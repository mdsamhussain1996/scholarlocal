from __future__ import annotations

import logging
import os
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from src.config import load_config
from src.index.bm25_store import BM25Store
from src.index.hybrid import HybridRetriever
from src.index.vector_store import VectorStore
from src.ingest.chunker import chunk_document
from src.ingest.metadata import extract_metadata_from_text, match_bibtex, parse_bib_files
from src.ingest.parser import compute_file_hash, parse_pdf
from src.ingest.theorems import extract_theorems_from_pages
from src.llm import OllamaClient
from src.storage import Database

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("scholarlocal")

app = typer.Typer(
    name="scholar",
    help="ScholarLocal: Private, offline research-paper agent for mathematics and control theory.",
    add_completion=False,
)
console = Console()


@app.command("status")
def status_cmd(config_path: str = "config.yaml"):
    """Check Ollama connectivity, hardware tier, and database statistics."""
    cfg = load_config(config_path)
    client = OllamaClient(
        host=cfg.models.ollama_host,
        chat_model=cfg.models.chat_model,
        embedding_model=cfg.models.embedding_model,
    )
    db = Database(os.path.join(cfg.data_dir, "scholar.db"))
    vstore = VectorStore(
        persist_dir=os.path.join(cfg.data_dir, "chroma"),
        ollama_client=client,
    )

    is_online = client.is_online()
    papers = db.list_papers()
    chunks = db.get_all_chunks()

    console.print(
        Panel.fit(
            f"[bold cyan]ScholarLocal Status[/bold cyan]\n"
            f"[bold]Hardware Tier:[/bold] {cfg.hardware_tier}\n"
            f"[bold]Chat Model:[/bold] {cfg.models.chat_model}\n"
            f"[bold]Embedding Model:[/bold] {cfg.models.embedding_model}\n"
            f"[bold]Ollama Daemon:[/bold] {'[green]Online (localhost:11434)[/green]' if is_online else '[red]Offline[/red]'}\n"
            f"[bold]Indexed Papers:[/bold] {len(papers)}\n"
            f"[bold]Indexed Chunks:[/bold] {len(chunks)}\n"
            f"[bold]ChromaDB Vectors:[/bold] {vstore.count()}\n"
            f"[bold]arXiv Network Sync:[/bold] {'[green]Enabled[/green]' if cfg.arxiv.enabled else '[yellow]Disabled (strictly offline)[/yellow]'}",
            title="System Diagnostics",
            border_style="blue",
        )
    )


@app.command("index")
def index_cmd(
    config_path: str = "config.yaml",
    force: bool = typer.Option(False, "--force", "-f", help="Force re-indexing of all PDFs even if unchanged."),
):
    """Index all PDFs in the library directory with exact pages, BibTeX keys, and vector embeddings."""
    cfg = load_config(config_path)
    console.print("[bold blue]Starting ScholarLocal Library Ingestion Pipeline...[/bold blue]")

    # Setup directories
    os.makedirs(cfg.data_dir, exist_ok=True)
    os.makedirs(cfg.library_dir, exist_ok=True)

    db = Database(os.path.join(cfg.data_dir, "scholar.db"))
    client = OllamaClient(
        host=cfg.models.ollama_host,
        chat_model=cfg.models.chat_model,
        embedding_model=cfg.models.embedding_model,
    )
    vstore = VectorStore(
        persist_dir=os.path.join(cfg.data_dir, "chroma"),
        ollama_client=client,
    )

    # 1. Parse BibTeX files
    bib_entries = parse_bib_files(cfg.bib_files)
    console.print(f"Loaded [bold]{len(bib_entries)}[/bold] reference entries from BibTeX files: {cfg.bib_files}")

    # 2. Discover PDFs
    pdf_files: list[str] = []
    for root, _, files in os.walk(cfg.library_dir):
        for f in files:
            if f.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, f))

    if not pdf_files:
        console.print(f"[yellow]No PDF files found in {cfg.library_dir}.[/yellow]")
        return

    console.print(f"Found [bold]{len(pdf_files)}[/bold] PDF document(s) in [cyan]{cfg.library_dir}[/cyan]")

    table = Table(title="Ingestion Summary", show_header=True, header_style="bold magenta")
    table.add_column("Paper / File", style="cyan")
    table.add_column("Cite Key", style="green")
    table.add_column("Pages", justify="right")
    table.add_column("Chunks", justify="right")
    table.add_column("Theorems", justify="right")
    table.add_column("Status", style="yellow")

    processed_count = 0
    skipped_count = 0

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        paper_id = os.path.splitext(filename)[0]
        file_hash = compute_file_hash(pdf_path)

        # Check existing
        existing_paper = db.get_paper_by_hash(file_hash)
        if existing_paper and not force:
            chunks = db.get_chunks_by_paper(existing_paper["id"])
            table.add_row(
                filename,
                existing_paper.get("cite_key") or "[dim]unmatched[/dim]",
                str(existing_paper.get("page_count", 0)),
                str(len(chunks)),
                "—",
                "[dim green]Skipped (unchanged)[/dim green]",
            )
            skipped_count += 1
            continue

        console.print(f"  --> Parsing & embedding [bold]{filename}[/bold]...")

        # Parse PDF
        pdf_doc = parse_pdf(pdf_path)
        page1_text = pdf_doc.pages[0][1] if pdf_doc.pages else ""

        # Extract metadata
        meta = extract_metadata_from_text(page1_text, client=client)
        if not meta.get("title") and pdf_doc.metadata.get("title"):
            meta["title"] = pdf_doc.metadata["title"]

        # Match with BibTeX
        matched_bib = match_bibtex(
            title=meta.get("title", ""),
            doi=meta.get("doi", ""),
            bib_entries=bib_entries,
            threshold=cfg.ingest.fuzzy_match_threshold,
        )
        cite_key = matched_bib["cite_key"] if matched_bib else ""

        # Use bibtex fields to supplement metadata if matched
        if matched_bib:
            meta["title"] = matched_bib.get("title") or meta["title"]
            meta["authors"] = matched_bib.get("author") or meta["authors"]
            meta["year"] = matched_bib.get("year") or meta["year"]
            meta["journal"] = matched_bib.get("journal") or meta["journal"]

        # Chunk document
        chunks = chunk_document(
            paper_id=paper_id,
            pages=pdf_doc.pages,
            target_tokens=cfg.ingest.chunk_size_tokens,
            overlap_tokens=cfg.ingest.chunk_overlap_tokens,
        )

        # Extract structured theorems
        theorems = extract_theorems_from_pages(
            paper_id=paper_id,
            pages=pdf_doc.pages,
            client=client,
            summarize_with_llm=False,  # Fast rule-based during bulk index
        )

        # Save to SQLite
        paper_record = {
            "id": paper_id,
            "file_path": pdf_path,
            "file_hash": file_hash,
            "title": meta.get("title", filename),
            "authors": meta.get("authors", ""),
            "year": meta.get("year"),
            "doi": meta.get("doi", ""),
            "journal": meta.get("journal", ""),
            "cite_key": cite_key,
            "abstract": meta.get("abstract", ""),
            "page_count": pdf_doc.page_count,
        }
        db.save_paper(paper_record)
        db.save_chunks(paper_id, chunks)

        # Save theorems to SQLite
        with db._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM theorems WHERE paper_id = ?", (paper_id,))
            for thm in theorems:
                cur.execute(
                    """
                    INSERT INTO theorems (id, paper_id, type, number, page, full_text, summary_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (thm["id"], paper_id, thm["type"], thm["number"], thm["page"], thm["full_text"], thm["summary_json"]),
                )
            conn.commit()

        # Save to Vector Store
        vstore.delete_paper_chunks(paper_id)
        vstore.add_chunks(chunks, paper_record)

        table.add_row(
            filename,
            cite_key if cite_key else "[red]unmatched[/red]",
            str(pdf_doc.page_count),
            str(len(chunks)),
            str(len(theorems)),
            "[bold green]Indexed[/bold green]",
        )
        processed_count += 1

    # 3. Build & save BM25 keyword index on all chunks
    all_chunks = db.get_all_chunks()
    papers_dict = {p["id"]: p for p in db.list_papers()}
    for c in all_chunks:
        p = papers_dict.get(c["paper_id"], {})
        c["cite_key"] = p.get("cite_key", "")
        c["year"] = p.get("year")
        c["title"] = p.get("title", "")
        c["authors"] = p.get("authors", "")

    bm25_store = BM25Store(os.path.join(cfg.data_dir, "bm25_index.pkl"))
    bm25_store.build_index(all_chunks)

    console.print(table)
    console.print(
        f"[bold green]Ingestion complete![/bold green] "
        f"Processed: {processed_count}, Unchanged: {skipped_count}, "
        f"Total in vector index: {vstore.count()} chunks, BM25 corpus: {len(all_chunks)} chunks."
    )


@app.command("search")
def search_cmd(
    query: str = typer.Argument(..., help="Search query string"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of chunks to return"),
    mode: str = typer.Option("hybrid", "--mode", "-m", help="Search mode: hybrid, vector, or bm25"),
    paper: Optional[str] = typer.Option(None, "--paper", "-p", help="Filter by paper ID or cite_key"),
    year_min: Optional[int] = typer.Option(None, "--year-min", help="Minimum publication year"),
    year_max: Optional[int] = typer.Option(None, "--year-max", help="Maximum publication year"),
    config_path: str = "config.yaml",
):
    """Perform hybrid (BM25 + vector RRF) or plain search with page-accurate citations."""
    cfg = load_config(config_path)
    client = OllamaClient(
        host=cfg.models.ollama_host,
        chat_model=cfg.models.chat_model,
        embedding_model=cfg.models.embedding_model,
    )
    vstore = VectorStore(
        persist_dir=os.path.join(cfg.data_dir, "chroma"),
        ollama_client=client,
    )
    bm25 = BM25Store(persist_path=os.path.join(cfg.data_dir, "bm25_index.pkl"))
    db = Database(os.path.join(cfg.data_dir, "scholar.db"))

    retriever = HybridRetriever(vector_store=vstore, bm25_store=bm25, db=db, config=cfg)

    # Build filters
    filters: dict[str, Any] = {}
    if paper:
        # Check if paper matches paper_id or cite_key
        filters["paper_id"] = paper
    if year_min:
        filters["year_min"] = year_min
    if year_max:
        filters["year_max"] = year_max

    mode_label = mode.upper()
    filter_str = f" [filters: {filters}]" if filters else ""
    console.print(f"[bold cyan]Query ({mode_label}):[/bold cyan] '{query}' (top {top_k}){filter_str}")

    results = retriever.search(query=query, top_k=top_k, filters=filters, mode=mode.lower())

    if not results:
        console.print("[yellow]No matching passages found for this query/filter.[/yellow]")
        return

    for idx, match in enumerate(results, start=1):
        citation = match.get("citation") or "[Unknown]"
        cite_key = match.get("cite_key") or "unknown"
        title = match.get("title") or match.get("paper_id")
        sec = match.get("section_title", "General")
        escaped_sec = escape(sec[:40])

        score_info = ""
        if mode == "hybrid":
            rrf = match.get("rrf_score", 0.0)
            vr = match.get("vector_rank")
            br = match.get("bm25_rank")
            score_info = f"RRF: {rrf:.5f} (Vec Rank: {vr or '—'}, BM25 Rank: {br or '—'})"
        elif mode == "vector":
            score_info = f"Vector Cosine Sim: {match.get('score', 0.0):.4f}"
        elif mode == "bm25":
            score_info = f"BM25 Score: {match.get('bm25_score', 0.0):.4f}"

        header = f"#{idx} | {escape(citation)} | {escape(cite_key)} | {score_info} | Sec: {escaped_sec}"
        console.print(Panel(
            f"[bold white]{escape(title)}[/bold white]\n\n"
            f"{escape(match['text'])}",
            title=f"[bold green]{header}[/bold green]",
            border_style="cyan" if idx == 1 else "blue",
        ))


if __name__ == "__main__":
    app()
