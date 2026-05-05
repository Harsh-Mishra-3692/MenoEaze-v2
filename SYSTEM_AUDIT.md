# SYSTEM AUDIT - MenoEaze (V5)

## 1. Modules
- **api.py**: FastAPI entry point for all ML/RAG operations.
- **pipeline.py**: Main orchestrator for feature building, inference, and RAG fusion.
- **sap_gru_model.py**: Core SAP-GRU architecture for longitudinal symptom prediction.
- **rag_engine.py**: Retrieval-Augmented Generation logic for grounded clinical advice.
- **hybrid_retriever.py**: Multi-stage vector and BM25 document retrieval.
- **db_client.py**: Supabase interface for data persistence and weights management.
- **memory.py**: User-level state management for temporal signal extraction.
- **personalization_adapter.py**: Dynamic adaptation layer for individual user weighting.
- **build_features.py**: Feature engineering pipeline (11 baseline to 56 temporal features).
- **trust_filter.py**: Reliability scoring based on user feedback history.

## 2. Core Pipeline
`input (Symptom Log)` → `build_features (56 features)` → `GRU (Severity Prediction)` → `Personalization (Adapter Weights)` → `RAG (Context Retrieval)` → `LLM (Grounded Reasoning)` → `output (Analysis + Action)`

## 3. Top 5 Risks
1. **Memory Volatility**: User session state in `memory.py` is volatile and requires database persistence.
2. **Grounding Drift**: Risk of LLM hallucination if retrieved clinical context is weak or missing.
3. **API Redundancy**: Potential for frontend to bypass unified `/run` endpoint for legacy logic.
4. **Schema Lock**: Rigid dependency between ML training schema and inference pipeline.
5. **Trust Signaling**: UI lacks explicit transparency for confidence scores and historical trends.

## 4. Protected Zones
- **ML Core**: `sap_gru_model.py`, `sap_gru.pt` (Weights and architecture).
- **Pipeline**: `pipeline.py`, `build_features.py` (Temporal logic and engineering).
- **API Contract**: `POST /run` request/response schema.
