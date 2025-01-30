-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the documentation chunks table
CREATE TABLE IF NOT EXISTS odoo_docs (
    id bigserial primary key,
    url varchar not null,
    chunk_number integer not null,
    version integer not null,
    title varchar not null,
    content text not null,
    metadata jsonb not null default '{}'::jsonb,
    embedding vector(768),
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    unique(url, chunk_number, version)
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_odoo_docs_version ON odoo_docs (version);
CREATE INDEX IF NOT EXISTS idx_odoo_docs_embedding ON odoo_docs
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 328);
CREATE INDEX IF NOT EXISTS idx_odoo_docs_metadata ON odoo_docs 
USING gin (metadata);

-- Create search function
CREATE OR REPLACE FUNCTION search_odoo_docs(
    query_embedding vector(768),
    version_num integer,
    match_limit integer
)
RETURNS TABLE (
    url character varying,
    title character varying,
    content text,
    similarity double precision
) 
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        d.url,
        d.title,
        d.content,
        (1 - (d.embedding <=> query_embedding)) AS similarity
    FROM odoo_docs d
    WHERE d.version = version_num
    ORDER BY 1 - (d.embedding <=> query_embedding) DESC
    LIMIT match_limit;
END;
$$;