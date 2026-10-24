"""A2A 0.3 JSON-RPC adapter backed by the same durable job queue as the UI.

Non-streaming, async tasks; message/send returns immediately. Clients poll tasks/get.
The adapter intentionally advertises no unsupported streaming or push capabilities.
"""
import base64
import uuid
from custom_agents.news import NewsRequest


def card(settings):
    return {'protocolVersion':'0.3.0', 'name':'News Writer', 'description':'Research sources, draft an original cited news article, and create an editorial cover with local models.',
        'url':settings.a2a_url, 'version':'0.1.0', 'preferredTransport':'JSONRPC',
        'capabilities':{'streaming':False,'pushNotifications':False},
        'defaultInputModes':['text/plain'], 'defaultOutputModes':['text/plain','application/json','image/png'],
        'securitySchemes':{'bearer':{'type':'http','scheme':'bearer'}}, 'security':[{'bearer':[]}],
        'skills':[{'id':'news-writing','name':'News research and writing','description':'Search and read sources, write a cited draft, and generate a cover. Defaults to Traditional Chinese for Hong Kong.',
            'tags':['news','research','writing'], 'examples':['近期開源 AI／本地模型新聞']}]}


def task(row, settings):
    state = {'queued':'submitted','running':'working','completed':'completed','failed':'failed'}[row['status']]
    status = {'state':state, 'timestamp':row['updated_at'].isoformat(),
        'message':{'role':'agent','messageId':str(uuid.uuid4()),'parts':[{'kind':'text','text':row['error'] or row['stage']}]}}
    result = {'kind':'task','id':str(row['id']),'contextId':str(row['id']),'status':status,
        'metadata':{'stage':row['stage'],'webUrl':settings.public_url+'/?job='+str(row['id'])}}
    if row['result']:
        r=row['result']
        text=r['article']['title']+'\n\n'+r['article']['body']+'\n\n'+ '\n'.join(f"[{s['id']}] {s['url']}" for s in r['sources'])
        result['artifacts']=[{'artifactId':'article','name':'article','parts':[{'kind':'text','text':text},{'kind':'data','data':r}]}]
        if r.get('cover'):
            # Machine clients receive the image through the authenticated gateway;
            # the studio file URL uses browser SSO and is unsuitable for API clients.
            cover_path=settings.data_dir/str(row['id'])/'cover.png'
            cover_file={'name':'cover.png','mimeType':'image/png'}
            if cover_path.is_file():
                cover_file['bytes']=base64.b64encode(cover_path.read_bytes()).decode('ascii')
            else:
                cover_file['uri']=settings.public_url+'/jobs/'+str(row['id'])+'/files/cover.png'
            result['artifacts'].append({'artifactId':'cover','name':'AI-generated editorial illustration',
                'parts':[{'kind':'file','file':cover_file}]})
    return result


async def dispatch(body, store, settings):
    request_id = body.get('id')
    def error(code, message):
        return {'jsonrpc':'2.0','id':request_id,'error':{'code':code,'message':message}}
    if body.get('jsonrpc') != '2.0' or not isinstance(body.get('params'), dict):
        return error(-32600,'Invalid JSON-RPC request')
    try:
        method, params = body.get('method'), body['params']
        if method == 'message/send':
            message=params['message']
            if message.get('role') != 'user' or not message.get('messageId'):
                return error(-32602,'Provide a user message with a messageId')
            text='\n'.join(p['text'] for p in message['parts'] if p.get('kind')=='text')
            request = NewsRequest.model_validate(params.get('metadata',{}).get('news_request') or {'topic':text})
            row=await store.submit(request.model_dump(), 'a2a:'+str(message['messageId'])[:200])
        elif method == 'tasks/get':
            row=await store.get(uuid.UUID(params['id']))
            if not row:
                return error(-32001,'Task not found')
        else:
            return error(-32601,'Method not supported; use message/send or tasks/get')
        return {'jsonrpc':'2.0','id':request_id,'result':task(row,settings)}
    except (ValueError, TypeError, KeyError):
        return error(-32602,'Invalid parameters or conflicting messageId')
