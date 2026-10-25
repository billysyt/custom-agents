"""Persistent article/source corpus. Exact search is sufficient for a personal newsroom."""
from psycopg.types.json import Jsonb
from agent_core.dedup import MODEL_VERSION,fingerprint,vector_literal

SCHEMA='''
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS news_documents (
 id bigserial PRIMARY KEY,
 kind text NOT NULL CHECK(kind IN ('article','source')),
 document_key text NOT NULL, job_id uuid, title text NOT NULL,
 content text NOT NULL, content_hash text NOT NULL,
 metadata jsonb NOT NULL DEFAULT '{}', model_version text NOT NULL,
 embedding vector(1024) NOT NULL, created_at timestamptz NOT NULL DEFAULT now(),
 UNIQUE(kind,document_key,model_version)
);
CREATE INDEX IF NOT EXISTS news_documents_jobs ON news_documents(job_id);
CREATE INDEX IF NOT EXISTS news_documents_hash ON news_documents(content_hash);
'''


class Archive:
    def __init__(self,store):self.store=store

    async def setup(self):
        async with self.store.pool.connection() as c:await c.execute(SCHEMA)

    async def put(self,kind,key,job_id,title,content,vector,metadata=None):
        async with self.store.pool.connection() as c:
            await c.execute('''INSERT INTO news_documents(kind,document_key,job_id,title,content,content_hash,metadata,model_version,embedding)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s::vector)
                ON CONFLICT(kind,document_key,model_version) DO UPDATE SET content=EXCLUDED.content,
                content_hash=EXCLUDED.content_hash,metadata=EXCLUDED.metadata,embedding=EXCLUDED.embedding''',
                (kind,key,job_id,title,content,fingerprint(content),Jsonb(metadata or {}),MODEL_VERSION,vector_literal(vector)))

    async def nearest(self,vector,kind='article',exclude_job=None,limit=5):
        async with self.store.pool.connection() as c:
            return await (await c.execute('''SELECT id,job_id,title,content,metadata,
                1-(embedding <=> %s::vector) AS similarity FROM news_documents
                WHERE kind=%s AND model_version=%s AND (%s::uuid IS NULL OR job_id IS DISTINCT FROM %s::uuid)
                ORDER BY embedding <=> %s::vector LIMIT %s''',
                (vector_literal(vector),kind,MODEL_VERSION,exclude_job,exclude_job,vector_literal(vector),limit))).fetchall()

    async def backfill_rows(self):
        async with self.store.pool.connection() as c:
            return await (await c.execute('''SELECT j.id,j.result FROM jobs j WHERE j.status='completed'
                AND j.result ? 'article' AND NOT EXISTS(SELECT 1 FROM news_documents d WHERE d.kind='article'
                AND d.job_id=j.id AND d.model_version=%s) ORDER BY j.created_at LIMIT 100''',(MODEL_VERSION,))).fetchall()

    async def stats(self):
        async with self.store.pool.connection() as c:
            return await (await c.execute('SELECT kind,count(*) AS count FROM news_documents GROUP BY kind')).fetchall()
