import socket
from types import SimpleNamespace
from datetime import datetime,timezone
import uuid
import pytest
import httpx
from custom_agents.a2a import dispatch, task
from custom_agents.news import NewsRequest, NewsWorkflow, article_length, validate_citations
from agent_core.web import public_url, resolve_public, Web

@pytest.mark.parametrize('url',['file:///etc/passwd','http://user:pw@example.com/','https://example.com:8443/'])
def test_reject_unsafe_urls(url):
    with pytest.raises(ValueError):public_url(url)

@pytest.mark.parametrize('ip',['127.0.0.1','10.0.0.1','169.254.169.254','::1','192.168.1.1'])
async def test_dns_private_addresses_rejected(monkeypatch,ip):
    async def fake(*a,**k):return [(socket.AF_INET,socket.SOCK_STREAM,6,'',(ip,443))]
    monkeypatch.setattr('asyncio.base_events.BaseEventLoop.getaddrinfo',fake)
    with pytest.raises(ValueError):await resolve_public('https://example.com/')

def test_citation_integrity():
    sources=[{'id':1},{'id':2}]
    validate_citations({'body':'A [1] and B [2]'},sources)
    for text in ['No citation','Only [1]','Invented [1] [3]','[1][2] https://invented.test']:
        with pytest.raises(ValueError):validate_citations({'body':text},sources)

def test_mixed_chinese_length_excludes_citations_and_model_names():
    assert article_length('本地 AI 模型 Qwen3.6，支援 262K context。[1][23]', '繁體中文（香港用語）') == (6, 'Chinese characters')
    assert article_length('Open-weight models run locally. [1][2]', 'English') == (4, 'words')

def test_length_revision_gives_direction_and_only_relevant_evidence():
    request=NewsRequest().model_dump()
    short={'body':'新聞'*200+'[1][2]'}
    instruction=NewsWorkflow.length_instruction(short,request)
    assert 'EXPAND' in instruction and '1.88 times' in instruction and '每段約 125 個漢字' in instruction
    assert 'SHORTEN' in NewsWorkflow.length_instruction({'body':'文'*1200},request)
    sources=[{'id':1,'text':'a'},{'id':2,'text':'b'},{'id':3,'text':'unrelated'}]
    import json
    evidence=json.loads(NewsWorkflow.revision_evidence({'request':request,'sources':sources,'brief':'unrelated'},short))
    assert evidence['sources']==sources[:2] and 'brief' not in evidence

async def test_review_repairs_oversized_draft_and_refuses_unbounded_output():
    workflow=object.__new__(NewsWorkflow)
    async def stage(*args):pass
    workflow.stage=stage
    oversized={'title':'Example title','body':'新聞內容'*300+'[1][2]','cover_prompt':'An editorial illustration'}
    valid={**oversized,'body':'新聞內容'*180+'[1][2]'}
    state={'job_id':'example','request':NewsRequest().model_dump(),'sources':[{'id':1},{'id':2}],
        'brief':'Verified sources','article':oversized,'timings':{}}
    calls=[]
    async def repair(system,data):
        calls.append(data)
        return oversized if len(calls)==1 else valid
    workflow.article_call=repair
    result=await workflow.review(state)
    assert result['article']==valid and len(calls)==2
    assert '1200 Chinese characters' in calls[1] and '600–900' in calls[1]
    calls.clear()
    async def never_fits(*args):calls.append(1);return oversized
    workflow.article_call=never_fits
    with pytest.raises(ValueError,match='Article length'):
        await workflow.review(state)
    assert len(calls)==3

async def test_reader_cannot_read_unsearched_url():
    web=Web('http://search:8080')
    try:assert 'error' in await web.read('https://example.com/')
    finally:await web.close()

async def test_redirect_cannot_escape_public_web(monkeypatch):
    web=Web('http://search:8080');web.allowed_urls.add('https://example.com/')
    async def resolve(url):
        if '127.0.0.1' in url:raise ValueError('private')
        return url,'93.184.216.34'
    monkeypatch.setattr('agent_core.web.resolve_public',resolve)
    await web.reader.aclose()
    web.reader=httpx.AsyncClient(transport=httpx.MockTransport(lambda r:httpx.Response(302,headers={'location':'http://127.0.0.1/admin'})))
    try:
        result=await web.read('https://example.com/')
        assert 'error' in result and web.sources==[]
    finally:await web.close()

class MemoryStore:
    def __init__(self):self.rows={}
    async def submit(self,request,key):
        if key not in self.rows:
            now=datetime.now(timezone.utc)
            self.rows[key]={'id':uuid.uuid4(),'request':request,'status':'queued','stage':'queued','error':None,'result':None,'updated_at':now}
        elif self.rows[key]['request']!=request:raise ValueError()
        return self.rows[key]
    async def get(self,id):return next((r for r in self.rows.values() if r['id']==id),None)

async def test_a2a_async_task_and_message_deduplication():
    store=MemoryStore();settings=SimpleNamespace(public_url='https://example.com/agents/news')
    body={'jsonrpc':'2.0','id':'r1','method':'message/send','params':{'message':{'role':'user','messageId':'unique','parts':[{'kind':'text','text':'Local model news'}]}}}
    a=await dispatch(body,store,settings);b=await dispatch(body,store,settings)
    assert a['result']['id']==b['result']['id']
    assert a['result']['status']['state']=='submitted'
    poll=await dispatch({'jsonrpc':'2.0','id':'r2','method':'tasks/get','params':{'id':a['result']['id']}},store,settings)
    assert poll['id']=='r2' and poll['result']['id']==a['result']['id']
    body['params']['message']['parts'][0]['text']='different topic'
    assert (await dispatch(body,store,settings))['error']['code']==-32602

async def test_auth_and_csrf(tmp_path):
    from custom_agents.app import create_app
    settings=SimpleNamespace(database_url='postgresql://unused',api_key='secret',trust_sso=True,public_url='https://example.com/agents/news',a2a_url='http://test/a2a')
    app=create_app(settings)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
        assert (await client.get('/agents/news/')).status_code==401
        assert (await client.get('/agents/news/',headers={'Authorization':'Bearer wrong'})).status_code==401
        assert (await client.get('/agents/news/',headers={'Authorization':'Bearer secret'})).status_code==200
        assert (await client.post('/agents/news/jobs',json={},headers={'X-authentik-username':'owner','Origin':'https://evil.test'})).status_code==403
        assert (await client.get('/health/live')).json()=={'status':'ok'}
        # LiteLLM resolves discovery relative to the configured JSON-RPC URL.
        for path in ['/.well-known/agent-card.json','/a2a/.well-known/agent-card.json']:
            response=await client.get(path)
            assert response.status_code==200 and response.json()['url']=='http://test/a2a'

def test_completed_a2a_cover_is_accessible_without_browser_sso(tmp_path):
    import base64
    job_id=uuid.uuid4();directory=tmp_path/str(job_id);directory.mkdir()
    png=b'\x89PNG\r\n\x1a\nexample';(directory/'cover.png').write_bytes(png)
    row={'id':job_id,'status':'completed','stage':'completed','updated_at':datetime.now(timezone.utc),
        'error':None,'result':{'article':{'title':'Example','body':'News'},'sources':[],'cover':'cover.png'}}
    result=task(row,SimpleNamespace(public_url='https://example.com/agents/news',data_dir=tmp_path))
    file=result['artifacts'][1]['parts'][0]['file']
    assert base64.b64decode(file['bytes'])==png and 'uri' not in file
