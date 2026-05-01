# MenoEaze Codebase Master Map

> **Operational Status:** Strict Read-Only Mode  
> **System Acknowledgment:** As an Elite Principal ML Architect, I am operating under zero-interference constraints. The mathematical logic, database structures, and ML engines are finalized and treated as sacred. This document serves exclusively as an architectural synthesis and academic mapping guide.

---

## 1. Architectural Synthesis ("What We Are Building")

MenoEaze is an **Adaptive Clinical Intelligence System**. It transcends simple heuristic wrapping by establishing a closed-loop learning architecture that pairs Temporal Deep Learning with a State-Conditioned Hybrid RAG pipeline.

The architecture relies on a **2-Layer GRU (Gated Recurrent Unit)** to forecast longitudinal symptom severity from an 11-dimensional clinical feature matrix. To overcome the high inter-patient variance inherent in physiological data, the system utilizes **Model-Agnostic Meta-Learning (MAML)**. This allows the GRU to perform *fast online adaptation* (few-shot learning) to an individual patient's baseline without suffering from catastrophic forgetting.

Simultaneously, the generative guidance layer utilizes a **State-Conditioned Hybrid RAG** engine. It dynamically retrieves context using both dense vector embeddings (Sentence-Transformers) and sparse lexical matching (BM25) via a Weighted Score Fusion algorithm. Generative safety is mathematically bounded by variance-based epistemic uncertainty quantification ($C = 1 / (1 + \sigma^2)$) and LLM-as-a-Judge semantic grounding checks, preventing medical hallucination.

---

## 2. Exhaustive Codebase Walkthrough

The repository follows a strict separation of concerns, decoupling the Next.js presentation layer from the FastAPI intelligence engine.

### Presentation & Interaction Layer (Next.js)
The frontend captures clinical data and orchestrates authentication, securely proxying LLM requests to the backend.
- `app/page.tsx`: The primary landing interface, heavily optimized for mobile-first user conversion.
- `app/dashboard/page.tsx`: The personalized analytics dashboard rendering severity forecasts and historical symptom metrics.
- `app/api/assistant/route.ts`: Secure Next.js Edge/Server route acting as a proxy for the LLM interaction layer, preventing `GROQ_API_KEY` leaks to the client.
- `components/SymptomForm.tsx`: A robust, mobile-optimized (44px touch targets) multi-step accordion form capturing the 11 clinical features required by the GRU model.
- `components/AnalysisCard.tsx`: Visual component displaying the model's inference outputs (base vs. adapted severity) and MAML activation status.
- `lib/ml/types.ts`: Centralized TypeScript definitions representing the (5, 11) tensor matrices required for data payload validation.

### Adaptive ML Backend (FastAPI / `ml_engine`)
The core intelligence layer executing temporal forecasting and vector retrieval.

#### A. Prediction & Deep Learning (Temporal GRU)
- `ml_engine/api.py`: The production FastAPI orchestrator. Exposes `/predict` and `/feedback` endpoints, managing the end-to-end inference flow.
- `ml_engine/model_def.py`: The single source of truth defining the PyTorch GRU architecture (Hidden=64, Dropout=0.2).
- `ml_engine/pipeline.py`: Assembles the base prediction, bias adjustment, adaptation, and gating logic into a unified inference function.
- `ml_engine/preprocess.py`: Performs deterministic min-max scaling, bounding all 11 features into a normalized `[0, 1]` tensor space.

#### B. Continual Learning (MAML & Online Adaptation)
- `ml_engine/meta_train.py`: Offline script handling the Model-Agnostic Meta-Learning outer loop, producing the generalized `meta_model.pt` weights.
- `ml_engine/maml_utils.py`: Utilities for calculating query/support loss during the meta-training inner loop.
- `ml_engine/adaptation.py`: The live online adaptation script. Updates the base model's final fully connected layer weights dynamically based on streaming patient feedback.
- `ml_engine/bias_control.py`: Enforces physiological constraints, clamping adaptive bias to prevent unbounded divergence (e.g., stopping the severity from exceeding realistic minimums/maximums).

#### C. Retrieval-Augmented Generation (Hybrid RAG)
- `ml_engine/rag_engine.py`: Controls the chunking, embedding, and inference prompt framing for the Groq LLM context window.
- `ml_engine/hybrid_retriever.py` & `ml_engine/reranker.py`: Coordinates the dual-path retrieval pipeline (Dense + Sparse).
- `ml_engine/db_client.py`: Interfaces with the localized ChromaDB vector store and Supabase for cloud metadata mapping.

#### D. Generative Safety & Telemetry (`ml_engine/research/`)
- `fusion.py`: Implements Reciprocal Rank Fusion (RRF) and Weighted Score Fusion algorithms.
- `uncertainty.py`: Quantifies epistemic uncertainty via variance measurement and entropy scaling, triggering safety fallbacks.
- `semantic_metrics.py` & `llm_judge.py`: Validates the generated response against context using cosine similarity and automated evaluation.

---

## 3. Current State ("Where We Are")

1. **Production Hardened**: Both the Next.js presentation boundary and the FastAPI inference engine are sanitized. All API keys are successfully isolated. Mobile UI targets are standardized.
2. **Mathematical Validation**: The MAML temporal architecture is fully integrated. In-memory diagnostic executions have empirically verified a **68.01% MSE variance reduction** following inner-loop adaptation.
3. **Generative Safety Verified**: The RAG pipeline exhibits extreme groundedness (>0.94) and an isolated hallucination rate of barely **2%**, heavily safeguarded by variance-bounded fallback triggers.
4. **Current Phase**: We are explicitly engaged in academic telemetry extraction and manuscript synthesis, transitioning the codebase's empirical proofs into an IEEE/Scopus-indexed research paper.

---

## 4. Academic Mapping (Paper Structure)

To facilitate the writing process for the research manuscript, map the architectural components directly to their respective academic sections.

### Section: Implementation & Architecture (The "How")
When detailing the system's structural design and mathematical logic, cite the following sources:
- **Temporal Modeling (GRU)**: Reference `model_def.py` and `pipeline.py` to describe the 11-feature tensor sequencing and dropout regularization.
- **Meta-Learning Optimization (MAML)**: Reference `meta_train.py` (outer loop optimization) and `adaptation.py` (online fast-adaptation).
- **Hybrid Document Retrieval**: Reference `fusion.py` to outline the exact weighting parameters ($\alpha=0.7$, $\beta=0.3$) and `bm25_index.py` for lexical sparsity mechanics.
- **Epistemic Uncertainty Quantification**: Reference `uncertainty.py` to detail the inverse variance-to-confidence calculation ($C = 1 / (1 + \sigma^2)$) governing edge-case hallucination prevention.

### Section: Result Analysis & Discussion (The "Proof")
When detailing the empirical efficacy, performance metrics, and ablation studies, pull raw data directly from these generated artifacts:
- **`ml_engine/results.txt`**: Contains the baseline GRU MAE and MSE statistics prior to personalization.
- **`ml_engine/training_history.csv`**: Demonstrates the epoch-over-epoch convergence rates and validation loss curves.
- **`ml_engine/research/empiric_audit_results.md`**: Provides the definitive quantitative impact table comparing the Global Baseline against MAML Adaptation (highlighting the 68% $\Delta$ MSE reduction).
- **`ml_engine/research/rag_performance_audit.md`**: Delivers the exact nDCG@5 metrics, Groundedness scores, and Hallucination benchmarks verifying generative safety.
- **Visual Assets**: Reference `pred_vs_actual.png`, `correlation.png`, and `residuals.png` to visually reinforce feature covariance and temporal trajectory alignments. Reference the custom graphviz-style architectural blocks (`bm25_fusion_diagram.png`) generated by `generate_paper_diagrams.py`.
