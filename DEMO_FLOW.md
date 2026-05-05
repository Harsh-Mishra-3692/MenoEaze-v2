# DEMO FLOW

1. **User Input**: User submits symptoms via frontend.
2. **Persistence**: Saved to Supabase DB (`symptom_logs`).
3. **Inference**: ML Engine computes severity via frozen SAP-GRU.
4. **DB Memory**: `memory.py` dynamically computes signal and trend directly from `predictions` table.
5. **RAG Retrieval**: Hybrid search uses semantic chunks, `top_k=5`, and `score > 0.15`.
6. **Grounded QA**: Assistant responds using strictly retrieved context or "Insufficient clinical evidence."
7. **Frontend View**: `AnalysisCard` natively renders real severity, trend, and confidence metrics without mock logic.
