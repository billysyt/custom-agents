import asyncio
from contextlib import asynccontextmanager, suppress
import hmac
import logging
from pathlib import Path
import uuid
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from agent_core.config import Settings
from agent_core.storage import Store
from custom_agents.news import NewsRequest, NewsWorkflow
from custom_agents.a2a import card, dispatch

LOG = logging.getLogger('custom-agents')
STAGES = {'queued','research','draft','review','cover','save','completed'}

async def worker(store, workflow):
    while True:
        try:
            # Dedicated pooled connection holds the lock for the entire worker session.
            async with store.pool.connection() as lock:
                acquired=await (await lock.execute('SELECT pg_try_advisory_lock(743119) AS held')).fetchone()
                if not acquired['held']:
                    await asyncio.sleep(5)
                    continue
                try:
                    await store.recover()
                    while True:
                        await lock.execute('SELECT 1')
                        row=await store.claim()
                        if not row:
                            await asyncio.sleep(2)
                            continue
                        async def monitor():
                            while True:
                                await asyncio.sleep(10)
                                await lock.execute('SELECT 1')
                        run=asyncio.create_task(workflow.run(row))
                        health=asyncio.create_task(monitor())
                        try:
                            async with asyncio.timeout(2400):
                                done,_=await asyncio.wait([run,health], return_when=asyncio.FIRST_COMPLETED)
                                if health in done:
                                    health.result()  # lost lock connection: cancel run and recover later
                                result=await run
                            await store.update(row['id'], status='completed', stage='completed', result=result, error=None)
                        except asyncio.CancelledError:
                            raise
                        except Exception as exc:
                            # Log exception type only: provider error bodies can contain credentials/prompts.
                            LOG.error('Job %s failed (%s)', row['id'], type(exc).__name__)
                            message = str(exc) if isinstance(exc, ValueError) else '暫時未能完成，請重試；已完成階段會保留。'
                            await store.update(row['id'], status='failed', error=message[:350])
                        finally:
                            for future in [run,health]:
                                future.cancel()
                                with suppress(asyncio.CancelledError, Exception):
                                    await future
                finally:
                    await lock.execute('SELECT pg_advisory_unlock(743119)')
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            LOG.error('Worker reconnecting (%s)',type(exc).__name__)
            await asyncio.sleep(5)


def create_app(settings=None):
    settings=settings or Settings()
    store=Store(settings.database_url)
    @asynccontextmanager
    async def lifespan(app):
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        await store.open()
        checkpoint=AsyncPostgresSaver(store.pool)
        await checkpoint.setup()
        workflow=NewsWorkflow(settings,store,checkpoint)
        process=asyncio.create_task(worker(store,workflow))
        app.state.worker=process
        try:
            yield
        finally:
            process.cancel()
            with suppress(asyncio.CancelledError):
                await process
            await store.close()
    app=FastAPI(title='Custom Agents', lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.store=store
    app.state.settings=settings

    @app.middleware('http')
    async def auth(request, call_next):
        path=request.url.path
        if path in {'/health/live','/health/ready','/.well-known/agent-card.json','/.well-known/agent.json'}:
            return await call_next(request)
        auth=request.headers.get('Authorization','')
        bearer=auth.startswith('Bearer ') and hmac.compare_digest(auth[7:],settings.api_key)
        sso=settings.trust_sso and bool(request.headers.get('X-authentik-username')) and path.startswith('/agents/news')
        if not (bearer or sso):
            return JSONResponse({'error':'Authentication required'},status_code=401)
        if sso and not bearer and request.method not in {'GET','HEAD','OPTIONS'}:
            expected=urlsplit(settings.public_url)
            origin=urlsplit(request.headers.get('Origin',''))
            if (origin.scheme,origin.netloc)!=(expected.scheme,expected.netloc):
                return JSONResponse({'error':'Invalid request origin'},status_code=403)
        if request.headers.get('content-length','0').isdigit() and int(request.headers.get('content-length','0')) > 32768:
            return JSONResponse({'error':'Request too large'},status_code=413)
        response=await call_next(request)
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Cache-Control']='no-store'
        response.headers['Content-Security-Policy']="default-src 'self'; img-src 'self'; style-src 'self'; script-src 'self'; frame-ancestors 'none'; base-uri 'self'"
        return response

    @app.get('/health/live')
    async def live():
        return {'status':'ok'}
    @app.get('/health/ready')
    async def ready():
        await store.healthy()
        if app.state.worker.done():
            raise HTTPException(503,'Worker unavailable')
        return {'status':'ok','database':'connected'}
    @app.get('/.well-known/agent-card.json')
    @app.get('/.well-known/agent.json')
    async def agent_card():
        return card(settings)
    @app.post('/a2a')
    async def rpc(request: Request):
        try:
            body=await request.json()
            if not isinstance(body,dict):
                raise ValueError()
        except ValueError:
            return {'jsonrpc':'2.0','id':None,'error':{'code':-32700,'message':'Parse error'}}
        return await dispatch(body,store,settings)

    @app.get('/agents/news')
    async def redirect():
        return RedirectResponse('/agents/news/')
    @app.get('/agents/news/')
    async def index():
        return FileResponse(Path(__file__).parent/'static/index.html')
    @app.get('/agents/news/jobs')
    async def jobs():
        return await store.list()
    @app.post('/agents/news/jobs',status_code=202)
    async def submit(body: NewsRequest, request: Request):
        key=request.headers.get('Idempotency-Key')
        if key and len(key)>200:
            raise HTTPException(400,'Idempotency key too long')
        try:
            return await store.submit(body.model_dump(),key)
        except ValueError as exc:
            raise HTTPException(409,str(exc))
    @app.get('/agents/news/jobs/{job_id}')
    async def get(job_id: uuid.UUID):
        row=await store.get(job_id)
        if not row:
            raise HTTPException(404,'Job not found')
        return row
    @app.post('/agents/news/jobs/{job_id}/retry',status_code=202)
    async def retry(job_id: uuid.UUID):
        row=await store.retry(job_id)
        if not row:
            raise HTTPException(409,'Only failed jobs can be retried')
        return row
    @app.get('/agents/news/jobs/{job_id}/files/{filename}')
    async def artifact(job_id: uuid.UUID,filename: str):
        if filename not in {'article.md','result.json','cover.png'} or not await store.get(job_id):
            raise HTTPException(404,'File not found')
        path=settings.data_dir/str(job_id)/filename
        if not path.is_file():
            raise HTTPException(404,'File not ready')
        return FileResponse(path,media_type={'cover.png':'image/png','article.md':'text/markdown','result.json':'application/json'}[filename],
            filename=filename if filename!='cover.png' else None)
    app.mount('/agents/news/static',StaticFiles(directory=Path(__file__).parent/'static'),name='static')
    return app
