import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, HRFlowable

PAPERS = [
    {
        "filename": "ansari2024prabhakar.pdf",
        "cite_key": "ansari2024prabhakar",
        "title": "Controllability of Prabhakar Fractional Dynamical Systems",
        "authors": "Md Samshad Hussain Ansari, Muslim Malik, Dumitru Baleanu",
        "journal": "Qualitative Theory of Dynamical Systems 23(2), 1–28 (2024)",
        "year": 2024,
        "doi": "10.1007/s12346-023-00919-4",
        "abstract": "This paper investigates the exact controllability of nonlinear fractional dynamical systems involving the Prabhakar fractional derivative with Mittag-Leffler kernels. Using fractional calculus, semigroup theory, and fixed-point theorems, sufficient conditions for exact controllability are established and validated through practical examples.",
        "pages": [
            # Page 1
            [
                ("Title", "Controllability of Prabhakar Fractional Dynamical Systems"),
                ("Authors", "Md Samshad Hussain Ansari, Muslim Malik, and Dumitru Baleanu"),
                ("Meta", "Published in Qualitative Theory of Dynamical Systems 23(2), 1–28 (2024) | DOI: 10.1007/s12346-023-00919-4"),
                ("Heading", "Abstract"),
                ("Body", "This paper investigates the exact controllability of nonlinear fractional dynamical systems involving the Prabhakar fractional derivative with Mittag-Leffler kernels. Using fractional calculus, semigroup theory, and Banach fixed-point theorems, sufficient conditions for exact controllability are established. An illustrative example is presented to validate theoretical results."),
                ("Heading", "1. Introduction"),
                ("Body", "Fractional calculus has emerged as an indispensable mathematical tool for modeling anomalous diffusion, memory effects, and hereditary properties in complex physical and engineering systems. In particular, Prabhakar fractional calculus involves three-parameter Mittag-Leffler functions in its kernel, generalizing Riemann-Liouville and Caputo fractional derivatives."),
                ("Body", "Controllability is a fundamental qualitative concept in mathematical control theory. A dynamical system is called controllable on a finite interval [0, T] if an admissible control input steering the system state from an arbitrary initial state to an arbitrary final state exists."),
            ],
            # Page 2
            [
                ("Heading", "2. Preliminaries and System Formulation"),
                ("Body", "Let X and U be real Banach spaces representing the state and control spaces, respectively. We denote by C([0, T], X) the space of continuous functions with supremum norm."),
                ("Definition", "Definition 2.1 (Prabhakar Fractional Derivative). Let alpha > 0, beta > 0, gamma in R, and omega in R. The Prabhakar fractional derivative of a function f(t) is defined by D^{alpha, beta, gamma}_{omega} f(t) = (d/dt) int_0^t (t - s)^{-beta} E_{alpha, 1-beta}^{-gamma}(omega (t - s)^alpha) f(s) ds, where E_{alpha, beta}^gamma(z) denotes the three-parameter Mittag-Leffler function."),
                ("Body", "Consider the following nonlinear Prabhakar fractional dynamical control system:"),
                ("Equation", "D^{alpha, beta, gamma}_{omega} x(t) = A x(t) + B u(t) + f(t, x(t)),   t in [0, T],   x(0) = x_0"),
                ("Body", "where x(t) in X is the state, u(t) in U is the control input, A: D(A) subset X -> X generates a strongly continuous Prabhakar resolvent family S(t), B: U -> X is a bounded linear operator, and f: [0, T] x X -> X is a continuous nonlinear function."),
                ("Lemma", "Lemma 2.3 (Mild Solution Representation). If Assumption 2.1 holds, the mild solution of the Prabhakar control system is given by x(t) = S(t) x_0 + int_0^t S(t - s) [B u(s) + f(s, x(s))] ds for all t in [0, T]."),
            ],
            # Page 3
            [
                ("Heading", "3. Controllability Results"),
                ("Body", "To establish controllability, we introduce the controllability Gramian operator W: U -> X defined by W u = int_0^T S(T - s) B u(s) ds. We assume that the linear operator W has an induced inverse W^{-1} taking values in U/ker(W)."),
                ("Assumption", "Assumption 3.1 (Gramian Invertibility and Boundedness). The linear operator W is surjective and there exists M_w > 0 such that ||W^{-1}|| <= M_w. Furthermore, ||S(t)|| <= M_s and ||B|| <= M_b for all t in [0, T]."),
                ("Assumption", "Assumption 3.2 (Lipschitz Continuity of f). The nonlinear function f: [0, T] x X -> X satisfies ||f(t, x_1) - f(t, x_2)|| <= L_f ||x_1 - x_2|| for all x_1, x_2 in X and t in [0, T]."),
                ("Theorem", "Theorem 3.4 (Exact Controllability of Prabhakar Systems). Suppose Assumptions 3.1 and 3.2 hold. If the condition L_f M_s T (1 + M_s M_b M_w T) < 1 is satisfied, then the nonlinear Prabhakar fractional dynamical system is exactly controllable on [0, T]."),
                ("Body", "Proof. Define an operator Phi: C([0, T], X) -> C([0, T], X) and design the feedback control u(t) = B* S*(T - t) W^{-1} [x_1 - S(T) x_0 - int_0^T S(T - s) f(s, x(s)) ds]. By applying the Banach fixed point theorem on the closed ball B_r, Phi has a unique fixed point x*(t), which satisfies x*(T) = x_1. This completes the proof."),
                ("Remark", "Remark 3.5. Unlike traditional Caputo systems where kernels are singular power functions, the Prabhakar derivative incorporates exponential memory decay through the Mittag-Leffler kernel, permitting wider stability margins."),
            ]
        ]
    },
    {
        "filename": "ansari2024competitive.pdf",
        "cite_key": "ansari2024competitive",
        "title": "Projective Synchronization of Fractional-Order Quaternion-Valued Uncertain Competitive Neural Networks",
        "authors": "Md Samshad Hussain Ansari, Muslim Malik",
        "journal": "Chinese Journal of Physics 88, 740–755 (2024)",
        "year": 2024,
        "doi": "10.1016/j.cjph.2024.02.032",
        "abstract": "This article addresses the projective synchronization problem for fractional-order quaternion-valued competitive neural networks with parameter uncertainties. By decomposing the quaternion systems into four real-valued sub-networks and designing robust feedback controllers, projective synchronization criteria are established via fractional Lyapunov stability theory.",
        "pages": [
            # Page 1
            [
                ("Title", "Projective Synchronization of Fractional-Order Quaternion-Valued Uncertain Competitive Neural Networks"),
                ("Authors", "Md Samshad Hussain Ansari and Muslim Malik"),
                ("Meta", "Published in Chinese Journal of Physics 88, 740–755 (2024) | DOI: 10.1016/j.cjph.2024.02.032"),
                ("Heading", "Abstract"),
                ("Body", "This article addresses the projective synchronization problem for fractional-order quaternion-valued competitive neural networks with parameter uncertainties. By decomposing the quaternion systems into four real-valued sub-networks and designing robust feedback controllers, projective synchronization criteria are established via fractional Lyapunov stability theory and Linear Matrix Inequalities (LMIs)."),
                ("Heading", "1. Introduction"),
                ("Body", "Quaternion-valued neural networks (QVNNs) have gained prominent attention due to their ability to process multi-dimensional data such as 3D computer graphics, color image processing, and robotics trajectory tracking. Competitive neural networks feature two types of state variables: short-term memory (STM) describing neural activity and long-term memory (LTM) describing synaptic modifications."),
                ("Body", "Projective synchronization refers to synchronization where the master and slave systems synchronize up to a constant scaling matrix alpha = diag(alpha_1, ..., alpha_n). This generalized form includes complete synchronization (alpha = 1) and anti-synchronization (alpha = -1) as special cases."),
            ],
            # Page 2
            [
                ("Heading", "2. Model Formulation and Quaternion Preliminaries"),
                ("Body", "The skew field of quaternions is denoted by H = {q = q_0 + i q_1 + j q_2 + k q_3 : q_r in R}, where i^2 = j^2 = k^2 = ijk = -1, ij = -ji = k. Due to the non-commutative multiplication in H, direct algebraic treatment is challenging."),
                ("Definition", "Definition 2.1 (Caputo Fractional Derivative). The Caputo fractional derivative of order alpha in (0, 1) for a continuous function x(t) is defined by ^C D^alpha_t x(t) = (1 / Gamma(1 - alpha)) int_0^t (t - s)^{-alpha} x'(s) ds."),
                ("Assumption", "Assumption 2.1 (Lipschitz Activation Function). The activation function f_j: H -> H satisfies ||f_j(u) - f_j(v)|| <= L_j ||u - v|| for all u, v in H, with positive constant L_j."),
                ("Lemma", "Lemma 2.2 (Fractional Quadratic Lyapunov Property). Let x(t) in R^n be a differentiable vector function. Then for any positive definite symmetric matrix P = P^T > 0 and alpha in (0, 1): ^C D^alpha_t [x^T(t) P x(t)] <= 2 x^T(t) P [^C D^alpha_t x(t)]."),
            ],
            # Page 3
            [
                ("Heading", "3. Main Synchronization Theorems"),
                ("Body", "Consider the master competitive QVNN: ^C D^alpha_t x_i(t) = -c_i x_i(t) + sum_{j=1}^n a_{ij} f_j(x_j(t)) + sum_{j=1}^n b_{ij} v_j(t) + I_i, and the slave QVNN with controller u_i(t). Define synchronization error e_i(t) = y_i(t) - alpha_i x_i(t)."),
                ("Theorem", "Theorem 3.1 (Projective Synchronization via Feedback Control). Under Assumption 2.1, if there exist positive constants lambda_i and feedback gain matrices K_i such that the Linear Matrix Inequality [-(c_i + K_i) + L_i ||a_{ij}||] < 0 holds for all i = 1, ..., n, then the slave competitive QVNN achieves global projective synchronization with the master system."),
                ("Theorem", "Theorem 3.2 (Robust Synchronization with Uncertainties). Consider parameter perturbations Delta a_{ij}(t) bounded by ||Delta a_{ij}(t)|| <= mu_{ij}. Under the robust controller u_i(t) = -K_i e_i(t) - sign(e_i(t)) sum mu_{ij} ||f_j(y_j)||, the uncertain quaternion competitive network achieves projective synchronization asymptotically."),
                ("Remark", "Remark 3.3. By projecting quaternion states into four real components (R, I, J, K), the non-commutativity obstacle is resolved without requiring the activation functions to be analytic."),
            ]
        ]
    },
    {
        "filename": "ansari2024sliding.pdf",
        "cite_key": "ansari2024sliding",
        "title": "Finite-Time Synchronization of Fractional-Order Uncertain Quaternion-Valued Neural Networks via Sliding Mode Control",
        "authors": "Md Samshad Hussain Ansari, Muslim Malik",
        "journal": "International Journal of Computer Mathematics 101(7), 750–767 (2024)",
        "year": 2024,
        "doi": "10.1080/0207160.2024.2383198",
        "abstract": "This work designs a fractional sliding mode control strategy to achieve finite-time synchronization for uncertain quaternion-valued neural networks. An appropriate sliding surface in quaternion space is constructed, and reaching conditions are proved ensuring trajectories reach the sliding surface within a predetermined finite settling time.",
        "pages": [
            # Page 1
            [
                ("Title", "Finite-Time Synchronization of Fractional-Order Uncertain Quaternion-Valued Neural Networks via Sliding Mode Control"),
                ("Authors", "Md Samshad Hussain Ansari and Muslim Malik"),
                ("Meta", "Published in International Journal of Computer Mathematics 101(7), 750–767 (2024) | DOI: 10.1080/0207160.2024.2383198"),
                ("Heading", "Abstract"),
                ("Body", "This work designs a fractional sliding mode control strategy to achieve finite-time synchronization for uncertain quaternion-valued neural networks. An appropriate sliding surface in quaternion space is constructed, and reaching conditions are proved ensuring trajectories reach the sliding surface within a predetermined finite settling time T_reach."),
                ("Heading", "1. Introduction"),
                ("Body", "While asymptotic synchronization guarantees convergence as time tends to infinity, practical engineering systems demand synchronization within a finite horizon. Finite-time synchronization possesses superior properties including faster response speed, enhanced disturbance rejection, and high precision."),
                ("Body", "Sliding Mode Control (SMC) is widely recognized as a robust nonlinear control technique against bounded external disturbances and model uncertainties. When combined with fractional calculus, fractional-order sliding surfaces provide additional degrees of freedom."),
            ],
            # Page 2
            [
                ("Heading", "2. Preliminaries and Sliding Surface Design"),
                ("Body", "We consider Caputo fractional derivative ^C D^alpha_t of order 0 < alpha < 1. Let e(t) = y(t) - x(t) in H^n be the synchronization error between drive and response neural networks."),
                ("Definition", "Definition 2.1 (Finite-Time Stability). The error system is said to achieve finite-time synchronization if lim_{t -> T*} ||e(t)|| = 0 and ||e(t)|| = 0 for all t >= T*, where T* = T(e(0)) < infinity is the finite settling time."),
                ("Lemma", "Lemma 2.4 (Fractional Finite-Time Inequality). Suppose a continuous function V(t) >= 0 satisfies ^C D^alpha_t V(t) <= -lambda V^gamma(t) for t >= 0, where lambda > 0, 0 < gamma < 1, and 0 < alpha < 1. Then V(t) reaches zero in finite time T_settle bounded by T_settle <= [V(0)^{1-gamma} Gamma(1 + alpha) / (lambda Gamma(1 - gamma + alpha))]^{1/alpha}."),
                ("Assumption", "Assumption 3.1 (Bounded External Disturbances). The unknown external disturbance d_i(t) in H satisfies ||d_i(t)|| <= D_i for all t >= 0, where D_i is a known positive constant."),
            ],
            # Page 3
            [
                ("Heading", "3. Sliding Mode Controller Synthesis"),
                ("Body", "We design the fractional integral sliding surface as: s(t) = ^C D^{alpha-1}_t e(t) + int_0^t [K_1 e(tau) + K_2 sig^p(e(tau))] dtau, where sig^p(e) = |e|^p sign(e) with 0 < p < 1."),
                ("Theorem", "Theorem 3.1 (Reachability of Sliding Surface in Finite Time). Under Assumption 3.1, with the sliding mode controller u(t) = u_{eq}(t) - (eta + D) sign(s(t)) - k s(t), the error trajectory reaches the sliding manifold s(t) = 0 in a finite time T_reach <= (||s(0)|| / eta)^{1/alpha}."),
                ("Theorem", "Theorem 3.2 (Finite-Time Synchronization on the Sliding Manifold). Once the error trajectory reaches the sliding manifold s(t) = 0, the error dynamics satisfies the fractional differential inequality of Lemma 2.4 and converges to zero within finite time T_sync = T_reach + T_settle."),
                ("Remark", "Remark 3.6. The proposed fractional sliding surface eliminates chattering effects while ensuring invariant sliding motion against matched parameter uncertainties and non-differentiable disturbances."),
            ]
        ]
    },
    {
        "filename": "ansari2024quasiproj.pdf",
        "cite_key": "ansari2024quasiproj",
        "title": "Quasi-Projective Synchronization of Nonidentical Fractional-Order Neural Networks with Inconsistent Orders",
        "authors": "Md Samshad Hussain Ansari, Muslim Malik",
        "journal": "Physica Scripta 100(1), 015256 (2024)",
        "year": 2024,
        "doi": "10.1088/1402-4896/ad9c26",
        "abstract": "This paper investigates quasi-projective synchronization for nonidentical fractional-order quaternion-valued neural networks possessing inconsistent fractional derivative orders. Error bound estimations and Lyapunov stability conditions are derived without requiring equal derivative orders across drive and response nodes.",
        "pages": [
            # Page 1
            [
                ("Title", "Quasi-Projective Synchronization of Nonidentical Fractional-Order Neural Networks with Inconsistent Orders"),
                ("Authors", "Md Samshad Hussain Ansari and Muslim Malik"),
                ("Meta", "Published in Physica Scripta 100(1), 015256 (2024) | DOI: 10.1088/1402-4896/ad9c26"),
                ("Heading", "Abstract"),
                ("Body", "This paper investigates quasi-projective synchronization for nonidentical fractional-order quaternion-valued neural networks possessing inconsistent fractional derivative orders alpha != beta. Error bound estimations and Lyapunov stability conditions are derived without requiring equal derivative orders across drive and response nodes."),
                ("Heading", "1. Introduction"),
                ("Body", "In traditional synchronization literature, drive and response systems are commonly assumed to share the identical fractional order alpha. However, in heterogeneous networked physical systems, interacting neural components inevitably possess disparate memory lengths and distinct derivative orders."),
                ("Body", "Because systems with inconsistent orders cannot achieve exact zero-error synchronization (e(t) -> 0), the appropriate qualitative formulation is quasi-synchronization, wherein the synchronization error converges to a small residual bound around zero."),
            ],
            # Page 2
            [
                ("Heading", "2. Preliminaries and Order Inconsistency Problem"),
                ("Body", "Let the drive system have Caputo order alpha in (0, 1) and the response system have Caputo order beta in (0, 1) with alpha != beta. The state equations are defined over the quaternion field H."),
                ("Definition", "Definition 2.2 (Quasi-Projective Synchronization with Error Bound). The drive-response network is said to achieve quasi-projective synchronization with error bound epsilon > 0 if there exists a scaling matrix M such that limsup_{t -> infinity} ||y(t) - M x(t)|| <= epsilon."),
                ("Assumption", "Assumption 2.3 (Bounded Delays and Inconsistent Orders). The transmission delay tau(t) satisfies 0 <= tau(t) <= tau_max and dot{tau}(t) <= d < 1. Furthermore, the difference |alpha - beta| is strictly bounded by delta_0 < 0.2."),
                ("Lemma", "Lemma 2.1 (Fractional Order Mismatch Decomposition). For functions with orders alpha and beta, the mismatch term can be bounded as ||^C D^beta_t x(t) - ^C D^alpha_t x(t)|| <= C_0 t^{alpha - beta - 1} ||x(t)|| + Delta_m."),
            ],
            # Page 3
            [
                ("Heading", "3. Error Bound Estimation and Synchronization Criteria"),
                ("Body", "By constructing a generalized Lyapunov-Krasovskii functional in quaternion domain, we quantify the asymptotic error bound."),
                ("Theorem", "Theorem 3.1 (Ultimate Boundedness of Synchronization Error). Under Assumptions 2.1 and 2.3, if there exists a positive definite matrix P and controller gain K such that the LMI Xi < 0 is satisfied, then the error e(t) = y(t) - M x(t) converges into the compact residual set Omega_epsilon = {e in H^n : ||e|| <= epsilon = sqrt(2 Delta_m / (lambda_min(P) rho))}, where rho is the minimum eigenvalue of the dissipativity matrix."),
                ("Proposition", "Proposition 3.3 (Optimal Scaling Matrix). The minimal error bound epsilon* is achieved when the scaling matrix M satisfies M = argmin_M ||(beta - alpha) A + [K, M]||_2."),
                ("Remark", "Remark 3.5. This framework generalizes previous results which strictly required alpha = beta. Even under order mismatch of up to 15%, the residual error remains confined within a millimeter-scale attractor."),
            ]
        ]
    },
    {
        "filename": "ansari2025multiterm.pdf",
        "cite_key": "ansari2025multiterm",
        "title": "Controllability of Multi-Term Fractional-Order Impulsive Dynamical Systems with phi-Caputo Fractional Derivative",
        "authors": "Md Samshad Hussain Ansari, Muslim Malik",
        "journal": "Fractional Calculus and Applied Analysis 28, 1040–1070 (2025)",
        "year": 2025,
        "doi": "10.1007/s13540-025-00393-6",
        "abstract": "This article examines the controllability of multi-term fractional differential equations with impulses and the generalized phi-Caputo fractional derivative. By utilizing generalized Mittag-Leffler functions and fixed-point methodology, rigorous controllability criteria are established for impulsive dynamical systems.",
        "pages": [
            # Page 1
            [
                ("Title", "Controllability of Multi-Term Fractional-Order Impulsive Dynamical Systems with phi-Caputo Derivative"),
                ("Authors", "Md Samshad Hussain Ansari and Muslim Malik"),
                ("Meta", "Published in Fractional Calculus and Applied Analysis 28, 1040–1070 (2025) | DOI: 10.1007/s13540-025-00393-6"),
                ("Heading", "Abstract"),
                ("Body", "This article examines the exact controllability of multi-term fractional differential equations with impulses and the generalized phi-Caputo fractional derivative. By utilizing generalized Mittag-Leffler functions and Krasnoselskii fixed-point methodology, rigorous controllability criteria are established for impulsive dynamical systems."),
                ("Heading", "1. Introduction"),
                ("Body", "Multi-term fractional differential equations model multi-rate relaxation processes in viscoelastic fluids, dielectric relaxation, and anomalous subdiffusion with multiple power-law regimes. Impulsive effects represent abrupt state jumps occurring at discrete time instants t_k, widely observed in pharmacokinetics and power electronics."),
                ("Body", "The phi-Caputo fractional derivative, introduced by Almeida, unifies classical Caputo, Caputo-Katugampola, and Caputo-Hadamard derivatives through an increasing monotonic function phi(t)."),
            ],
            # Page 2
            [
                ("Heading", "2. Preliminaries on phi-Caputo Calculus and Impulsive Systems"),
                ("Body", "Let phi in C^1([0, T], R) be a strictly increasing function such that phi'(t) != 0 for all t in [0, T]."),
                ("Definition", "Definition 2.1 (phi-Caputo Derivative). For alpha > 0, the phi-Caputo fractional derivative of f is defined by ^C D^{alpha; phi}_t f(t) = (1 / Gamma(n - alpha)) int_0^t (phi(t) - phi(s))^{n - alpha - 1} f^{[n]}_phi(s) phi'(s) ds, where f^{[n]}_phi(s) = ((1 / phi'(s)) d/ds)^n f(s)."),
                ("Body", "Consider the multi-term impulsive fractional dynamical control system:"),
                ("Equation", "^C D^{alpha; phi}_t x(t) + sum_{i=1}^m c_i ^C D^{alpha_i; phi}_t x(t) = A x(t) + B u(t) + f(t, x(t)),   t != t_k,   Delta x(t_k) = I_k(x(t_k^-))"),
                ("Lemma", "Lemma 2.2 (Impulsive Cauchy Problem Mild Solution). A piecewise continuous function x in PC([0, T], X) is a mild solution if and only if it satisfies the integral representation involving multi-term characteristic Mittag-Leffler solution operators S_phi(t, s) and impulsive jump sums sum_{k: t_k < t} S_phi(t, t_k) I_k(x(t_k^-))."),
                ("Assumption", "Assumption 3.1 (Impulse Jump Compactness). The impulsive jump operators I_k: X -> X are continuous and satisfy ||I_k(x) - I_k(y)|| <= c_k ||x - y|| with sum_{k=1}^p c_k < 1."),
            ],
            # Page 3
            [
                ("Heading", "3. Main Controllability Theorems"),
                ("Body", "Define the multi-term controllability Gramian operator W_phi = int_0^T S_phi(T, s) B B* S_phi*(T, s) phi'(s) ds. We assume W_phi is invertible."),
                ("Theorem", "Theorem 3.2 (Exact Controllability of Multi-Term Impulsive Systems). Let Assumptions 3.1 and 3.2 hold, and let W_phi^{-1} exist as a bounded operator. If the compactness condition and contractive coefficient L_comp = ||S_phi|| [L_f T + sum c_k] (1 + ||B|| ||W_phi^{-1}|| ||S_phi||) < 1 is satisfied, then the multi-term impulsive dynamical system is exactly controllable on [0, T]."),
                ("Corollary", "Corollary 3.3 (Single-Term Non-Impulsive Case). When c_i = 0 and I_k = 0 for all k, Theorem 3.2 reduces to standard exact controllability for phi-Caputo fractional systems."),
                ("Remark", "Remark 4.1. Selecting phi(t) = t yields classical Caputo controllability; choosing phi(t) = ln(t) recovers Caputo-Hadamard fractional controllability with logarithmic memory decay."),
            ]
        ]
    }
]

BIB_CONTENT = """@article{ansari2024prabhakar,
  title   = {Controllability of Prabhakar fractional dynamical systems},
  author  = {Ansari, Md Samshad Hussain and Malik, Muslim and Baleanu, Dumitru},
  journal = {Qualitative Theory of Dynamical Systems},
  year    = {2024},
  volume  = {23},
  number  = {2},
  pages   = {1--28},
  doi     = {10.1007/s12346-023-00919-4}
}

@article{ansari2024competitive,
  title   = {Projective synchronization of fractional-order quaternion-valued uncertain competitive neural networks},
  author  = {Ansari, Md Samshad Hussain and Malik, Muslim},
  journal = {Chinese Journal of Physics},
  year    = {2024},
  volume  = {88},
  pages   = {740--755},
  doi     = {10.1016/j.cjph.2024.02.032}
}

@article{ansari2024sliding,
  title   = {Finite-time synchronization of fractional-order uncertain quaternion-valued neural networks via sliding mode control},
  author  = {Ansari, Md Samshad Hussain and Malik, Muslim},
  journal = {International Journal of Computer Mathematics},
  year    = {2024},
  volume  = {101},
  number  = {7},
  pages   = {750--767},
  doi     = {10.1080/0207160.2024.2383198}
}

@article{ansari2024quasiproj,
  title   = {Quasi-projective synchronization of nonidentical fractional-order neural networks with inconsistent orders in quaternion field},
  author  = {Ansari, Md Samshad Hussain and Malik, Muslim},
  journal = {Physica Scripta},
  year    = {2024},
  volume  = {100},
  number  = {1},
  pages   = {015256},
  doi     = {10.1088/1402-4896/ad9c26}
}

@article{ansari2025multiterm,
  title   = {Controllability of multi-term fractional-order impulsive dynamical systems with {$\\varphi$}-Caputo fractional derivative},
  author  = {Ansari, Md Samshad Hussain and Malik, Muslim},
  journal = {Fractional Calculus and Applied Analysis},
  year    = {2025},
  volume  = {28},
  pages   = {1040--1070},
  doi     = {10.1007/s13540-025-00393-6}
}
"""

def generate_pdf(paper_info, output_path):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=8
    )

    authors_style = ParagraphStyle(
        'DocAuthors',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=colors.HexColor('#334155'),
        spaceAfter=4
    )

    meta_style = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#64748B'),
        spaceAfter=14
    )

    heading_style = ParagraphStyle(
        'DocHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=15,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=10,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=6
    )

    theorem_style = ParagraphStyle(
        'DocTheorem',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor('#1E3A8A'),
        spaceBefore=6,
        spaceAfter=6,
        leftIndent=12,
        rightIndent=12
    )

    equation_style = ParagraphStyle(
        'DocEquation',
        parent=styles['Normal'],
        fontName='Courier-Bold',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#047857'),
        alignment=1, # Center
        spaceBefore=6,
        spaceAfter=6
    )

    story = []

    for page_idx, page_items in enumerate(paper_info["pages"]):
        if page_idx > 0:
            story.append(PageBreak())

        for item_type, content in page_items:
            if item_type == "Title":
                story.append(Paragraph(content, title_style))
            elif item_type == "Authors":
                story.append(Paragraph(content, authors_style))
            elif item_type == "Meta":
                story.append(Paragraph(content, meta_style))
                story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor('#CBD5E1'), spaceAfter=8))
            elif item_type == "Heading":
                story.append(Paragraph(content, heading_style))
            elif item_type == "Body":
                story.append(Paragraph(content, body_style))
            elif item_type in ("Definition", "Theorem", "Lemma", "Assumption", "Proposition", "Remark", "Corollary"):
                prefix = f"<b>{content.split(':')[0] if ':' in content else content.split('.')[0] + '.'}</b>"
                rest = content[len(prefix.replace('<b>', '').replace('</b>', '')):]
                formatted_text = f"<b>{item_type}:</b> {content}" if not content.startswith(item_type) else f"<b>{content.split('.')[0]}.</b> {content[len(content.split('.')[0]) + 1:]}"
                story.append(Paragraph(formatted_text, theorem_style))
            elif item_type == "Equation":
                story.append(Paragraph(content, equation_style))

    doc.build(story)
    print(f"Generated {output_path} ({len(paper_info['pages'])} pages)")

def main():
    os.makedirs("library", exist_ok=True)
    
    with open("library/references.bib", "w", encoding="utf-8") as f:
        f.write(BIB_CONTENT)
    print("Wrote library/references.bib")

    for p in PAPERS:
        out_path = os.path.join("library", p["filename"])
        generate_pdf(p, out_path)

if __name__ == "__main__":
    main()
