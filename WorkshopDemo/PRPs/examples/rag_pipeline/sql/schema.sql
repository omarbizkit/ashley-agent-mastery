-- WorkshopDemo RAG Pipeline Schema
-- All objects prefixed with wd_ to avoid conflicts with existing database objects

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Drop existing WorkshopDemo objects if they exist
DROP TABLE IF EXISTS wd_chunks CASCADE;
DROP TABLE IF EXISTS wd_documents CASCADE;
DROP INDEX IF EXISTS idx_wd_chunks_embedding;
DROP INDEX IF EXISTS idx_wd_chunks_document_id;
DROP INDEX IF EXISTS idx_wd_documents_metadata;
DROP INDEX IF EXISTS idx_wd_chunks_content_trgm;

-- Documents table for storing source documents
CREATE TABLE wd_documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title TEXT NOT NULL,
    source TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_wd_documents_metadata ON wd_documents USING GIN (metadata);
CREATE INDEX idx_wd_documents_created_at ON wd_documents (created_at DESC);

-- Chunks table for storing document chunks with embeddings
CREATE TABLE wd_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES wd_documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding vector(1536),
    chunk_index INTEGER NOT NULL,
    metadata JSONB DEFAULT '{}',
    token_count INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_wd_chunks_embedding ON wd_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 1);
CREATE INDEX idx_wd_chunks_document_id ON wd_chunks (document_id);
CREATE INDEX idx_wd_chunks_chunk_index ON wd_chunks (document_id, chunk_index);
CREATE INDEX idx_wd_chunks_content_trgm ON wd_chunks USING GIN (content gin_trgm_ops);

-- Vector search function for pure semantic similarity
CREATE OR REPLACE FUNCTION wd_match_chunks(
    query_embedding vector(1536),
    match_count INT DEFAULT 10
)
RETURNS TABLE (
    chunk_id UUID,
    document_id UUID,
    content TEXT,
    similarity FLOAT,
    metadata JSONB,
    document_title TEXT,
    document_source TEXT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        c.id AS chunk_id,
        c.document_id,
        c.content,
        1 - (c.embedding <=> query_embedding) AS similarity,
        c.metadata,
        d.title AS document_title,
        d.source AS document_source
    FROM wd_chunks c
    JOIN wd_documents d ON c.document_id = d.id
    WHERE c.embedding IS NOT NULL
    ORDER BY c.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Hybrid search function combining vector + TSVector keyword search
CREATE OR REPLACE FUNCTION wd_hybrid_search(
    query_embedding vector(1536),
    query_text TEXT,
    match_count INT DEFAULT 10,
    text_weight FLOAT DEFAULT 0.3
)
RETURNS TABLE (
    chunk_id UUID,
    document_id UUID,
    content TEXT,
    combined_score FLOAT,
    vector_similarity FLOAT,
    text_similarity FLOAT,
    metadata JSONB,
    document_title TEXT,
    document_source TEXT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    WITH vector_results AS (
        SELECT
            c.id AS chunk_id,
            c.document_id,
            c.content,
            1 - (c.embedding <=> query_embedding) AS vector_sim,
            c.metadata,
            d.title AS doc_title,
            d.source AS doc_source
        FROM wd_chunks c
        JOIN wd_documents d ON c.document_id = d.id
        WHERE c.embedding IS NOT NULL
    ),
    text_results AS (
        SELECT
            c.id AS chunk_id,
            c.document_id,
            c.content,
            ts_rank_cd(to_tsvector('english', c.content), plainto_tsquery('english', query_text)) AS text_sim,
            c.metadata,
            d.title AS doc_title,
            d.source AS doc_source
        FROM wd_chunks c
        JOIN wd_documents d ON c.document_id = d.id
        WHERE to_tsvector('english', c.content) @@ plainto_tsquery('english', query_text)
    )
    SELECT
        COALESCE(v.chunk_id, t.chunk_id) AS chunk_id,
        COALESCE(v.document_id, t.document_id) AS document_id,
        COALESCE(v.content, t.content) AS content,
        (COALESCE(v.vector_sim, 0) * (1 - text_weight) + COALESCE(t.text_sim, 0) * text_weight) AS combined_score,
        COALESCE(v.vector_sim, 0) AS vector_similarity,
        COALESCE(t.text_sim, 0) AS text_similarity,
        COALESCE(v.metadata, t.metadata) AS metadata,
        COALESCE(v.doc_title, t.doc_title) AS document_title,
        COALESCE(v.doc_source, t.doc_source) AS document_source
    FROM vector_results v
    FULL OUTER JOIN text_results t ON v.chunk_id = t.chunk_id
    ORDER BY combined_score DESC
    LIMIT match_count;
END;
$$;

-- Get all chunks for a specific document
CREATE OR REPLACE FUNCTION wd_get_document_chunks(doc_id UUID)
RETURNS TABLE (
    chunk_id UUID,
    content TEXT,
    chunk_index INTEGER,
    metadata JSONB
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        id AS chunk_id,
        wd_chunks.content,
        wd_chunks.chunk_index,
        wd_chunks.metadata
    FROM wd_chunks
    WHERE document_id = doc_id
    ORDER BY chunk_index;
END;
$$;

-- Trigger function to auto-update updated_at column
CREATE OR REPLACE FUNCTION wd_update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for wd_documents table
DROP TRIGGER IF EXISTS wd_update_documents_updated_at ON wd_documents;
CREATE TRIGGER wd_update_documents_updated_at BEFORE UPDATE ON wd_documents
    FOR EACH ROW EXECUTE FUNCTION wd_update_updated_at_column();
