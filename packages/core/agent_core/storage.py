"""PostgreSQL queue. A session advisory lock fences the single GPU worker.

No in-memory queue is authoritative. Interrupted runs resume their LangGraph
checkpoint on the next worker start. Failed jobs require explicit retry.
"""
import hashlib
import json
import uuid
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
 id uuid PRIMARY KEY, request jsonb NOT NULL, status text NOT NULL DEFAULT 'queued',
 stage text NOT NULL DEFAULT 'queued', result jsonb, error text,
 idempotency_key text UNIQUE, attempts integer NOT NULL DEFAULT 0,
 created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS jobs_queue_idx ON jobs(status,created_at);
"""

class Store:
    def __init__(self, url):
        self.pool = AsyncConnectionPool(url, min_size=1, max_size=5, open=False,
            kwargs={'autocommit': True, 'row_factory': dict_row})

    async def open(self):
        await self.pool.open()
        async with self.pool.connection() as conn:
            await conn.execute(SCHEMA)

    async def close(self):
        await self.pool.close()

    async def submit(self, request, key=None):
        # Scoped to this single-owner service. Same key + different body is an error.
        async with self.pool.connection() as conn:
            row = await (await conn.execute("""INSERT INTO jobs(id,request,idempotency_key)
                VALUES(%s,%s,%s) ON CONFLICT(idempotency_key) DO NOTHING RETURNING *""",
                (uuid.uuid4(), Jsonb(request), key))).fetchone()
            if row is None:
                row = await (await conn.execute('SELECT * FROM jobs WHERE idempotency_key=%s', (key,))).fetchone()
                if row['request'] != request:
                    raise ValueError('Idempotency key already used with a different request')
            return row

    async def get(self, job_id):
        async with self.pool.connection() as conn:
            return await (await conn.execute('SELECT * FROM jobs WHERE id=%s', (job_id,))).fetchone()

    async def list(self):
        async with self.pool.connection() as conn:
            return await (await conn.execute('SELECT * FROM jobs ORDER BY created_at DESC LIMIT 50')).fetchall()

    async def update(self, job_id, **changes):
        allowed = {'status', 'stage', 'result', 'error'}
        if not changes or set(changes) - allowed:
            raise ValueError('Unsupported job change')
        keys = list(changes)
        values = [Jsonb(changes[k]) if k == 'result' and changes[k] is not None else changes[k] for k in keys]
        async with self.pool.connection() as conn:
            await conn.execute('UPDATE jobs SET ' + ','.join(k+'=%s' for k in keys) + ',updated_at=now() WHERE id=%s', (*values, job_id))

    async def claim(self):
        async with self.pool.connection() as conn:
            return await (await conn.execute("""UPDATE jobs SET status='running',attempts=attempts+1,
                updated_at=now() WHERE id=(SELECT id FROM jobs WHERE status='queued'
                ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1) RETURNING *""")).fetchone()

    async def retry(self, job_id):
        async with self.pool.connection() as conn:
            return await (await conn.execute("""UPDATE jobs SET status='queued',error=NULL,updated_at=now()
                WHERE id=%s AND status='failed' RETURNING *""", (job_id,))).fetchone()

    async def recover(self):
        # Called only while holding the worker advisory lock.
        async with self.pool.connection() as conn:
            await conn.execute("UPDATE jobs SET status='queued',updated_at=now() WHERE status='running'")

    async def healthy(self):
        async with self.pool.connection() as conn:
            await conn.execute('SELECT 1')
