-- ==========================================
-- MenoEaze Production Schema (Phase 3)
-- Strict Safety & RLS Enforcement
-- ==========================================

-- 1. USERS (Profiles linked to auth.users)
CREATE TABLE IF NOT EXISTS public.users (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    email TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 2. SYMPTOM LOGS
CREATE TABLE IF NOT EXISTS public.symptom_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
    -- Strict array constraint matching the ML engine requirements (11 features)
    feature_vector FLOAT[] NOT NULL CHECK (array_length(feature_vector, 1) = 11),
    notes TEXT,
    emoji VARCHAR(10),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 3. PREDICTIONS
CREATE TABLE IF NOT EXISTS public.predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
    severity FLOAT NOT NULL CHECK (severity >= 0 AND severity <= 1),
    confidence FLOAT NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    personalized BOOLEAN NOT NULL DEFAULT FALSE,
    trend VARCHAR(20) DEFAULT 'unknown',
    anomaly BOOLEAN NOT NULL DEFAULT FALSE,
    doctor_recommend BOOLEAN NOT NULL DEFAULT FALSE,
    doctor_urgency VARCHAR(20) DEFAULT 'none',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 4. FEEDBACK (Post-Inference Signal)
CREATE TABLE IF NOT EXISTS public.feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
    -- 1:1 relationship with a prediction
    prediction_id UUID NOT NULL UNIQUE REFERENCES public.predictions(id) ON DELETE CASCADE,
    predicted FLOAT NOT NULL CHECK (predicted >= 0 AND predicted <= 1),
    actual FLOAT NOT NULL CHECK (actual >= 0 AND actual <= 1),
    rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 10),
    trust_score FLOAT CHECK (trust_score >= 0 AND trust_score <= 1),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 5. TRUST SCORES (Historical Aggregation)
CREATE TABLE IF NOT EXISTS public.trust_scores (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL UNIQUE REFERENCES public.users(id) ON DELETE CASCADE,
    aggregate_trust FLOAT NOT NULL DEFAULT 0.5 CHECK (aggregate_trust >= 0 AND aggregate_trust <= 1),
    feedback_count INTEGER NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 6. USER WEIGHTS (Personalization Adapter)
CREATE TABLE IF NOT EXISTS public.user_weights (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL UNIQUE REFERENCES public.users(id) ON DELETE CASCADE,
    baseline_offset FLOAT NOT NULL DEFAULT 0.0,
    volatility_multiplier FLOAT NOT NULL DEFAULT 1.0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 7. USER MEMORY (Pre-computed Signals)
CREATE TABLE IF NOT EXISTS public.user_memory (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL UNIQUE REFERENCES public.users(id) ON DELETE CASCADE,
    avg_severity FLOAT NOT NULL DEFAULT 0.0 CHECK (avg_severity >= 0 AND avg_severity <= 1),
    historical_trend VARCHAR(20) DEFAULT 'unknown',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 8. RAG LOGS (Grounding Audit Trail)
CREATE TABLE IF NOT EXISTS public.rag_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
    prediction_id UUID REFERENCES public.predictions(id) ON DELETE CASCADE,
    query TEXT NOT NULL,
    answer TEXT NOT NULL,
    sources JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ==========================================
-- INDEXING (Performance & Time-Series)
-- ==========================================
CREATE INDEX IF NOT EXISTS idx_symptom_logs_user_id_time ON public.symptom_logs(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_user_id_time ON public.predictions(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_feedback_user_id_time ON public.feedback(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_rag_logs_user_id_time ON public.rag_logs(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_feedback_prediction_id ON public.feedback(prediction_id);

-- ==========================================
-- ROW LEVEL SECURITY (RLS) - CRITICAL
-- ==========================================
ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.symptom_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.feedback ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.trust_scores ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.user_weights ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.user_memory ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.rag_logs ENABLE ROW LEVEL SECURITY;

-- 100% Isolation: Users can ONLY interact with their own auth.uid() rows.
DROP POLICY IF EXISTS "Users can manage their own profile" ON public.users;
CREATE POLICY "Users can manage their own profile" ON public.users FOR ALL USING (auth.uid() = id);

DROP POLICY IF EXISTS "Users can manage their own symptom logs" ON public.symptom_logs;
CREATE POLICY "Users can manage their own symptom logs" ON public.symptom_logs FOR ALL USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can read their own predictions" ON public.predictions;
CREATE POLICY "Users can read their own predictions" ON public.predictions FOR ALL USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can read their own feedback" ON public.feedback;
CREATE POLICY "Users can read their own feedback" ON public.feedback FOR ALL USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can read their own trust scores" ON public.trust_scores;
CREATE POLICY "Users can read their own trust scores" ON public.trust_scores FOR ALL USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can read their own weights" ON public.user_weights;
CREATE POLICY "Users can read their own weights" ON public.user_weights FOR ALL USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can read their own memory" ON public.user_memory;
CREATE POLICY "Users can read their own memory" ON public.user_memory FOR ALL USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can read their own rag logs" ON public.rag_logs;
CREATE POLICY "Users can read their own rag logs" ON public.rag_logs FOR ALL USING (auth.uid() = user_id);

-- Revoke all public access as a final safety measure
REVOKE ALL ON ALL TABLES IN SCHEMA public FROM PUBLIC;
