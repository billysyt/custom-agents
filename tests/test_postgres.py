"""Real PostgreSQL durability tests; opt in with a disposable TEST_DATABASE_URL."""
import os
import uuid
import asyncio
import pytest
from agent_core.storage import Store
from langgraph.graph import StateGraph,START,END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from typing import TypedDict

pytestmark=pytest.mark.skipif(not os.getenv('TEST_DATABASE_URL'),reason='Needs a disposable PostgreSQL database')

async def test_queue_recovery_and_idempotency():
    store=Store(os.environ['TEST_DATABASE_URL']);await store.open()
    try:
        async with store.pool.connection() as c:await c.execute('TRUNCATE jobs')
        request={'topic':'test'}
        a=await store.submit(request,'stable-id');b=await store.submit(request,'stable-id')
        assert a['id']==b['id']
        with pytest.raises(ValueError):await store.submit({'topic':'other'},'stable-id')
        rows=await asyncio.gather(*[store.claim() for _ in range(5)])
        assert len([r for r in rows if r])==1
        await store.recover()
        row=await store.claim();assert row['id']==a['id'] and row['attempts']==2
        await store.update(row['id'],status='completed',result={'saved':True})
    finally:await store.close()
    reopened=Store(os.environ['TEST_DATABASE_URL']);await reopened.open()
    try:assert (await reopened.get(a['id']))['result']=={'saved':True}
    finally:await reopened.close()

async def test_checkpoint_resume_does_not_repeat_completed_stage():
    store=Store(os.environ['TEST_DATABASE_URL']);await store.open()
    calls=[];fail=[True]
    class S(TypedDict,total=False):one:str;two:str
    async def first(state):calls.append('first');return {'one':'saved'}
    async def second(state):
        calls.append('second')
        if fail[0]:raise RuntimeError('simulated interruption')
        return {'two':state['one']+' resumed'}
    try:
        saver=AsyncPostgresSaver(store.pool);await saver.setup()
        def build():
            g=StateGraph(S);g.add_node('first',first);g.add_node('second',second)
            g.add_edge(START,'first');g.add_edge('first','second');g.add_edge('second',END)
            return g.compile(checkpointer=saver)
        config={'configurable':{'thread_id':str(uuid.uuid4())}}
        with pytest.raises(RuntimeError):await build().ainvoke({},config)
        fail[0]=False
        result=await build().ainvoke(None,config)
        assert result['two']=='saved resumed'
        assert calls==['first','second','second']
    finally:await store.close()


async def test_vector_archive_survives_reopen_and_excludes_current_job():
    from agent_core.archive import Archive
    store=Store(os.environ['TEST_DATABASE_URL']);await store.open()
    key=str(uuid.uuid4());other=str(uuid.uuid4())
    a=[1.0]+[0.0]*1023;b=[0.0,1.0]+[0.0]*1022
    try:
        archive=Archive(store);await archive.setup()
        await archive.put('article',key,key,'Same event','content',a)
        await archive.put('article',key,key,'Same event','updated content',a)
        await archive.put('article',other,other,'Different event','other',b)
        rows=await archive.nearest(a,exclude_job=other)
        assert str(rows[0]['job_id'])==key and rows[0]['content']=='updated content'
        assert rows[0]['similarity']==pytest.approx(1.0)
        assert all(str(r['job_id'])!=key for r in await archive.nearest(a,exclude_job=key))
    finally:await store.close()
    store=Store(os.environ['TEST_DATABASE_URL']);await store.open()
    try:assert str((await Archive(store).nearest(a))[0]['job_id'])==key
    finally:await store.close()
