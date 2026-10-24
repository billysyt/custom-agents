import socket
from types import SimpleNamespace
from datetime import datetime,timezone
import uuid
import pytest
import httpx
from custom_agents.a2a import dispatch, task
from custom_agents.news import NewsRequest, validate_citations
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
    settings=SimpleNamespace(database_url='postgresql://unused',api_key='secret',trust_sso=True,public_url='https://example.com/agents/news')
    app=create_app(settings)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
        assert (await client.get('/agents/news/')).status_code==401
        assert (await client.get('/agents/news/',headers={'Authorization':'Bearer wrong'})).status_code==401
        assert (await client.get('/agents/news/',headers={'Authorization':'Bearer secret'})).status_code==200
        assert (await client.post('/agents/news/jobs',json={},headers={'X-authentik-username':'owner','Origin':'https://evil.test'})).status_code==403
        assert (await client.get('/health/live')).json()=={'status':'ok'}
