-- ==========================================
-- RAG Schema Extension (Phase 3.1)
-- ==========================================

-- Enable the pgvector extension to work with embeddings
CREATE EXTENSION IF NOT EXISTS vector;

-- 1. MEDICAL DOCUMENTS (RAG Sink)
CREATE TABLE IF NOT EXISTS public.medical_documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT,
    content TEXT NOT NULL,
    source TEXT,
    document_name TEXT,
    chunk_index INTEGER,
    embedding VECTOR(384), -- HF miniLM-L6-v2 dimensionality
    hash TEXT UNIQUE,
    priority FLOAT DEFAULT 0.0,
    is_deleted BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 2. VECTOR INDEX (Performance)
CREATE INDEX IF NOT EXISTS idx_medical_documents_embedding 
ON public.medical_documents 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- 3. MATCH FUNCTION (RPC)
CREATE OR REPLACE FUNCTION public.match_medical_documents (
  query_embedding VECTOR(384),
  match_threshold FLOAT DEFAULT 0.5,
  match_count INTEGER DEFAULT 5
)
RETURNS TABLE (
  id UUID,
  content TEXT,
  document_name TEXT,
  similarity FLOAT,
  priority FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    md.id,
    md.content,
    md.document_name,
    1 - (md.embedding <=> query_embedding) AS similarity,
    md.priority
  FROM public.medical_documents md
  WHERE md.is_deleted = FALSE
    AND 1 - (md.embedding <=> query_embedding) > match_threshold
  ORDER BY similarity DESC
  LIMIT match_count;
END;
$$;

-- 4. PERMISSIONS
ALTER TABLE public.medical_documents ENABLE ROW LEVEL SECURITY;

-- Anonymous users can read medical documents (for RAG query context)
CREATE POLICY "Public read access for medical documents" 
ON public.medical_documents FOR SELECT 
USING (TRUE);
