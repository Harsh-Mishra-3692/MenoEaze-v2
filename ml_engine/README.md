# MenoEaze ML Engine — Health Intelligence System

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Next.js Frontend (Vercel)                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────┐  │
│  │ Dashboard │  │ Symptom  │  │ Analysis │  │ Assistant │   │
│  │   Page   │  │   Form   │  │   Card   │  │   Chat    │   │
│  └────┬─────┘  └──────────┘  └────┬─────┘  └───────────┘  │
│       │                           │ HTTP                    │
└───────┼───────────────────────────┼─────────────────────────┘
        │                           │
        ▼                           ▼
┌─────────────────────────────────────────────────────────────┐
│              FastAPI Backend (ml_engine/api.py)              │
│                                                              │
│  POST /predict ─────────────────────────────────────────┐   │
│    │                                                     │   │
│    ├─ 1. Validate input (5, 11)                         │   │
│    ├─ 2. Apply scalers (scalers.pt)                     │   │
│    ├─ 3. Base prediction (model.pt / meta_model.pt)     │   │
│    ├─ 4. Online adaptation (1 step, fc-only, safe)      │   │
│    ├─ 5. Personalize (user_bias + clamp)                │   │
│    ├─ 6. RAG guidance (ChromaDB → Groq LLM)            │   │
│    └─ 7. Store & respond                                │   │
│                                                              │
│  POST /feedback ── store actual severity, invalidate cache  │
│  GET  /health ──── system status                            │
│                                                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│  │model_def │ │adaptation│ │rag_engine│ │user_store│       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘      │
└──────────────────────────┬──────────────┬───────────────────┘
                           │              │
                    ┌──────┴──────┐ ┌─────┴─────┐
                    │  ChromaDB   │ │  Supabase  │
                    │ (chroma_db/)│ │  (cloud)   │
                    │ 26 PDFs     │ │ user data  │
                    └─────────────┘ └────────────┘
```

## Components

| File | Purpose |
|------|---------|
| `model_def.py` | Shared GRU architecture (single source of truth) |
| `api.py` | Production FastAPI with full prediction pipeline |
| `adaptation.py` | Safe online adaptation (1 step, fc-only, bounded) |
| `rag_engine.py` | ChromaDB RAG with metadata-rich chunking + Groq LLM |
| `user_store.py` | Supabase data layer (30-day rolling buffer) |
| `maml_utils.py` | MAML inner/outer loop utilities |
| `meta_train.py` | Offline MAML meta-training script |
| `continual_train.py` | Offline continual learning from feedback |
| `ingest_pdfs.py` | PDF ingestion into ChromaDB |
| `test_pipeline.py` | End-to-end test suite |

**DO NOT MODIFY**: `generate_data.py`, `preprocess.py`, `train_gru.py`

## Quick Start

### 1. Install Dependencies

```bash
cd ml_engine
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Ingest PDFs into ChromaDB

```bash
python ingest_pdfs.py
# Use --force to re-ingest
```

### 4. (Optional) Run MAML Meta-Training

```bash
python meta_train.py
# Produces meta_model.pt (API auto-selects it if available)
```

### 5. Start the API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Run Tests

```bash
python test_pipeline.py
```

## API Reference

### `GET /health`
Returns system status.

### `POST /predict`
```json
{
  "user_id": "optional-uuid",
  "sequence": [[...], [...], [...], [...], [...]],
  "request_guidance": true
}
```

**sequence**: 5 rows × 11 features, all in [0,1]:
`[hot_flash, night_sweats, sleep, mood, fatigue, anxiety, activity, stress, caffeine, age, bmi]`

**Response**:
```json
{
  "severity": 0.4523,
  "base_severity": 0.4600,
  "adapted": true,
  "user_bias": -0.0077,
  "prediction_id": "uuid",
  "guidance": "According to the NAMS 2022 Position Statement...",
  "sources": [{"title": "NAMS 2022 Hormone Therapy Position Statement", "file": "..."}],
  "severity_level": "medium",
  "latency_ms": 45.2
}
```

### `POST /feedback`
```json
{
  "user_id": "uuid",
  "prediction_id": "uuid",
  "actual_severity": 0.65
}
```

## Supabase Schema

```sql
CREATE TABLE IF NOT EXISTS prediction_history (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID NOT NULL,
  input_sequence JSONB NOT NULL,
  predicted_severity FLOAT NOT NULL,
  adapted BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS user_feedback (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID NOT NULL,
  prediction_id UUID REFERENCES prediction_history(id),
  actual_severity FLOAT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW()
);
```

## Deployment

### Local Development
```bash
uvicorn api:app --port 8000 --reload
```

### Production
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1
```

> **Note**: Use `--workers 1` because the model is loaded in-process.
> For multi-worker setups, use a model server like TorchServe.

### Frontend Config
Add to your Next.js `.env.local`:
```
NEXT_PUBLIC_ML_API_URL=http://localhost:8000
```

For production, point to your deployed FastAPI URL.
