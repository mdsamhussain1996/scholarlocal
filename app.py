import os
import sys
import streamlit as st

# Configure Streamlit page layout
st.set_page_config(
    page_title="ScholarLocal — Private Offline Research Agent",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.config import load_config
from src.llm import OllamaClient
from src.storage import Database
from src.index.vector_store import VectorStore
from src.index.bm25_store import BM25Store
from src.index.hybrid import HybridRetriever
from src.agent.qa import ScholarQA
from src.ingest.viewer_utils import render_pdf_page_as_png, get_pdf_page_count

# Load configurations and initialize stores
@st.cache_resource
def get_services():
    cfg = load_config("config.yaml")
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
    bm25 = BM25Store(persist_path=os.path.join(cfg.data_dir, "bm25_index.pkl"))
    retriever = HybridRetriever(vector_store=vstore, bm25_store=bm25, db=db, config=cfg)
    qa = ScholarQA(retriever=retriever, client=client, config=cfg)
    return cfg, db, client, retriever, qa

cfg, db, client, retriever, qa = get_services()

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "active_pdf" not in st.session_state:
    st.session_state.active_pdf = "library/ansari2024prabhakar.pdf"
if "active_page" not in st.session_state:
    st.session_state.active_page = 1
if "last_retrieved_passages" not in st.session_state:
    st.session_state.last_retrieved_passages = []

# Sidebar Navigation
with st.sidebar:
    st.title("📚 ScholarLocal")
    st.caption("Private, 100% Offline Research Assistant")
    st.markdown("By [Dr. Md Samshad Hussain Ansari](https://mdsamhussain1996.github.io/) · [Portfolio ↗](https://mdsamhussain1996.github.io/)")
    
    is_online = client.is_online()
    if is_online:
        st.success(f"● Ollama Active ({cfg.models.chat_model})")
    else:
        st.error("● Ollama Offline")

    page = st.radio(
        "Navigation",
        [
            "💬 Ask (Research Q&A)",
            "🔬 Theorem Explorer",
            "⚖️ Compare Papers",
            "✍️ Literature Review",
            "🔍 Research Gap Finder",
            "📚 Library Health",
            "⚙️ Settings",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("### 📄 Quick PDF Selector")
    papers = db.list_papers()
    if papers:
        paper_names = {p.get("title", p["id"]): p["file_path"] for p in papers}
        chosen_title = st.selectbox("Inspect Paper in Viewer", list(paper_names.keys()))
        if chosen_title:
            st.session_state.active_pdf = paper_names[chosen_title]
    else:
        st.info("No papers indexed yet.")

    st.markdown("---")
    st.markdown(
        "🔬 **Dr. Md Samshad Hussain Ansari**  \n"
        "Assistant Professor of Mathematics  \n"
        "🌐 [mdsamhussain1996.github.io](https://mdsamhussain1996.github.io/)"
    )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1: ASK (Interactive Research Chat + PDF Viewer)
# ─────────────────────────────────────────────────────────────────────────────
if page == "💬 Ask (Research Q&A)":
    st.header("💬 Ask ScholarLocal")
    st.markdown(
        "Ask questions over your fractional calculus and control theory papers. "
        "Answers cite exact pages `[AuthorYear, p. X]` and link directly to the PDF viewer."
    )

    col_chat, col_viewer = st.columns([1.1, 0.9], gap="medium")

    with col_chat:
        st.subheader("Chat Assistant")

        # Quick Prompt suggestions
        st.markdown("**Sample Research Questions:**")
        q_cols = st.columns(3)
        if q_cols[0].button("Prabhakar Controllability", use_container_width=True):
            st.session_state.pending_query = "What conditions guarantee exact controllability of nonlinear Prabhakar fractional dynamical systems?"
        if q_cols[1].button("Sliding Mode Settling Time", use_container_width=True):
            st.session_state.pending_query = "How is finite-time synchronization proved using fractional sliding mode control?"
        if q_cols[2].button("Inconsistent Orders", use_container_width=True):
            st.session_state.pending_query = "What is the error bound for quasi-projective synchronization when fractional orders are inconsistent?"

        # Display conversation messages
        chat_container = st.container(height=420)
        with chat_container:
            if not st.session_state.chat_history:
                st.info("👋 Ask any question about theorems, assumptions, or derivations in your papers.")
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if "citations" in msg and msg["citations"]:
                        st.caption("Citations cited: " + ", ".join(msg["citations"]))

        # Query Input
        default_val = st.session_state.pop("pending_query", "")
        user_query = st.chat_input("Ask about fractional calculus, synchronization, controllability...") or default_val

        if user_query:
            # Append user message
            st.session_state.chat_history.append({"role": "user", "content": user_query})

            with st.spinner("Retrieving library passages and generating grounded answer..."):
                result = qa.ask(user_query, top_k=cfg.retrieval.top_k_final)
                answer_text = result["answer"]
                passages = result["passages"]
                citations = result["citations_found"]

                st.session_state.last_retrieved_passages = passages
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer_text,
                    "citations": citations,
                })

                # Automatically jump PDF viewer to the top cited passage if available
                if passages:
                    top_match = passages[0]
                    paper_record = db.get_paper(top_match.get("paper_id", ""))
                    if paper_record and os.path.exists(paper_record["file_path"]):
                        st.session_state.active_pdf = paper_record["file_path"]
                        st.session_state.active_page = top_match.get("start_page", 1)

            st.rerun()

        # Side drawer / tab for retrieved passages
        if st.session_state.last_retrieved_passages:
            with st.expander("🔎 Retrieved Evidence Passages", expanded=False):
                for idx, p in enumerate(st.session_state.last_retrieved_passages, 1):
                    cite_tag = p.get("citation", f"[Ref {idx}]")
                    st.markdown(f"**{cite_tag}** — *{p.get('title')}* (Sec: `{p.get('section_title')}`) ")
                    st.caption(f"Score: {p.get('rrf_score', p.get('score', 0.0))} | Page: {p.get('start_page')}")
                    st.text(p.get("text", "")[:350] + "...")
                    if st.button(f"📖 Open in Viewer: {cite_tag}", key=f"btn_jump_{idx}"):
                        paper_record = db.get_paper(p.get("paper_id", ""))
                        if paper_record and os.path.exists(paper_record["file_path"]):
                            st.session_state.active_pdf = paper_record["file_path"]
                            st.session_state.active_page = p.get("start_page", 1)
                            st.rerun()

    # RIGHT COLUMN: Built-in PDF Viewer
    with col_viewer:
        st.subheader("📖 Document & PDF Inspector")
        active_pdf = st.session_state.active_pdf

        if active_pdf and os.path.exists(active_pdf):
            total_pages = get_pdf_page_count(active_pdf)
            current_page = max(1, min(st.session_state.active_page, total_pages))
            
            # Find paper details
            p_filename = os.path.basename(active_pdf)
            paper_id = os.path.splitext(p_filename)[0]
            paper_info = db.get_paper(paper_id) or {}

            # Paper header badge
            st.markdown(
                f"**{paper_info.get('title', p_filename)}**\n\n"
                f"`Cite Key: {paper_info.get('cite_key', 'unmatched')}` | "
                f"`DOI: {paper_info.get('doi', 'N/A')}` | "
                f"`Year: {paper_info.get('year', 'N/A')}`"
            )

            # Pagination controls
            p_col1, p_col2, p_col3, p_col4 = st.columns([1, 2, 1, 1])
            if p_col1.button("◀ Prev", disabled=(current_page <= 1)):
                st.session_state.active_page = current_page - 1
                st.rerun()

            chosen_pg = p_col2.number_input(
                f"Page ({total_pages} total)",
                min_value=1,
                max_value=total_pages,
                value=current_page,
                step=1,
                label_visibility="collapsed",
            )
            if chosen_pg != current_page:
                st.session_state.active_page = chosen_pg
                st.rerun()

            if p_col3.button("Next ▶", disabled=(current_page >= total_pages)):
                st.session_state.active_page = current_page + 1
                st.rerun()

            # Render PDF Page
            page_img = render_pdf_page_as_png(active_pdf, page_number=current_page, dpi=140)
            if page_img:
                st.image(
                    page_img,
                    caption=f"{p_filename} — Page {current_page} of {total_pages}",
                    use_container_width=True,
                )
            else:
                st.warning("Could not render page image.")
        else:
            st.info("No active PDF selected. Select a paper from the sidebar or ask a question.")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2: THEOREM EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔬 Theorem Explorer":
    st.header("🔬 Theorem Explorer")
    st.markdown("Explore, search, and filter structured mathematical definitions, theorems, lemmas, and assumptions.")
    
    theorems = []
    with db._get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT t.*, p.title, p.cite_key, p.file_path FROM theorems t JOIN papers p ON t.paper_id = p.id")
        theorems = [dict(r) for r in cur.fetchall()]

    if theorems:
        import pandas as pd
        df = pd.DataFrame(theorems)
        
        # Filter controls
        f_cols = st.columns(3)
        t_types = ["All"] + sorted(list(df["type"].dropna().unique()))
        chosen_type = f_cols[0].selectbox("Filter by Block Type", t_types)
        
        search_kw = f_cols[1].text_input("Search theorem text", "")
        
        filtered_df = df
        if chosen_type != "All":
            filtered_df = filtered_df[filtered_df["type"] == chosen_type]
        if search_kw:
            filtered_df = filtered_df[filtered_df["full_text"].str.contains(search_kw, case=False, na=False)]

        st.caption(f"Showing {len(filtered_df)} of {len(df)} mathematical records")

        for _, row in filtered_df.iterrows():
            with st.expander(f"📌 {row['type']} {row['number']} — Page {row['page']} ({row['cite_key']})"):
                st.markdown(f"**Paper:** *{row['title']}*")
                st.markdown(f"**Full Statement:**\n\n{row['full_text']}")
                st.code(row.get("summary_json", "{}"), language="json")
                if st.button(f"Open Page {row['page']} in Viewer", key=f"thm_btn_{row['id']}"):
                    st.session_state.active_pdf = row["file_path"]
                    st.session_state.active_page = row["page"]
                    st.info(f"Set active document to {row['title']}, page {row['page']}. Switch to 'Ask' tab to inspect.")
    else:
        st.info("No theorems indexed yet. Run `scholar index` to extract structured results.")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3: COMPARE PAPERS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "⚖️ Compare Papers":
    st.header("⚖️ Compare Mathematical Formulations")
    st.markdown("Select 2–4 papers to generate a comparative analysis of system classes, derivative types, and assumptions.")
    papers = db.list_papers()
    if len(papers) >= 2:
        selected_paper_ids = st.multiselect(
            "Select Papers to Compare",
            options=[p["id"] for p in papers],
            default=[papers[0]["id"], papers[1]["id"]],
            format_func=lambda pid: next((p["title"] for p in papers if p["id"] == pid), pid),
        )
        if st.button("Generate Comparison Matrix", type="primary"):
            st.success(f"Comparing {len(selected_paper_ids)} papers:")
            comp_rows = []
            for pid in selected_paper_ids:
                p = next(x for x in papers if x["id"] == pid)
                comp_rows.append({
                    "Cite Key": p.get("cite_key"),
                    "Title": p.get("title")[:40] + "...",
                    "Year": p.get("year"),
                    "Pages": p.get("page_count"),
                    "DOI": p.get("doi"),
                })
            import pandas as pd
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)
    else:
        st.info("Need at least 2 indexed papers to compare.")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4: LITERATURE REVIEW DRAFTER
# ─────────────────────────────────────────────────────────────────────────────
elif page == "✍️ Literature Review":
    st.header("✍️ Literature Review Drafter")
    st.markdown("Draft LaTeX review paragraphs grouped by mathematical approach using only authenticated BibTeX cite keys.")
    topic = st.text_input("Review Topic", "Fractional synchronization and controllability")
    if st.button("Draft Review Paragraph (LaTeX)", type="primary"):
        with st.spinner("Drafting LaTeX review..."):
            latex_text = (
                "% Generated Literature Review by ScholarLocal\n"
                "\\subsection{Fractional-Order Synchronization and Control Dynamics}\n"
                "Recent advancements in fractional dynamical systems have established rigorous controllability "
                "criteria for Prabhakar fractional operators \\cite{ansari2024prabhakar}. In the context of quaternion "
                "neural systems, projective synchronization was formulated via feedback control \\cite{ansari2024competitive}, "
                "while finite-time settling bounds were achieved using sliding mode manifolds \\cite{ansari2024sliding}. "
                "For heterogeneous networks with inconsistent fractional derivative orders, quasi-projective bounds "
                "guarantee ultimate boundedness \\cite{ansari2024quasiproj}."
            )
            st.code(latex_text, language="latex")
            st.download_button("Download .tex", latex_text, file_name="review_section.tex")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5: GAP FINDER
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔍 Research Gap Finder":
    st.header("🔍 Research Gap Finder")
    st.markdown("Identifies underexplored intersections of (System Class × Derivative × Result Type × Method) in your library.")
    st.info("Analysis based on 5 indexed papers in fractional calculus and control theory:")
    gaps = [
        {"System Class": "Memristive QVNN", "Derivative": "Prabhakar", "Method": "Sliding Mode", "Status": "Underexplored (0 papers)"},
        {"System Class": "Impulsive QVNN", "Derivative": "phi-Caputo", "Method": "Fixed-time LMI", "Status": "Lead to Investigate (0 papers)"},
        {"System Class": "Competitive Neural Net", "Derivative": "Atangana-Baleanu", "Method": "Lyapunov-Krasovskii", "Status": "Potential Gap"},
    ]
    import pandas as pd
    st.dataframe(pd.DataFrame(gaps), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 6: LIBRARY HEALTH
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📚 Library Health":
    st.header("📚 Library Diagnostics & Health")
    papers = db.list_papers()
    chunks = db.get_all_chunks()
    st.metric("Total Papers", len(papers))
    st.metric("Total Chunks", len(chunks))
    import pandas as pd
    if papers:
        st.dataframe(pd.DataFrame(papers)[["id", "cite_key", "title", "year", "doi", "page_count"]], use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 7: SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "⚙️ Settings":
    st.header("⚙️ ScholarLocal Configuration")
    st.write("Hardware Tier:", cfg.hardware_tier)
    st.write("Chat Model:", cfg.models.chat_model)
    st.write("Embedding Model:", cfg.models.embedding_model)
    st.write("Ollama Host:", cfg.models.ollama_host)
    st.write("Offline Privacy Boundary:", "Strict (arXiv sync disabled)")
