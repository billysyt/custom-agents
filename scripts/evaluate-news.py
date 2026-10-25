#!/usr/bin/env python3
"""Run the actual BGE retrieval + Qwen novelty guard against a versioned dataset.

Set the same LiteLLM/Langfuse environment as the service. No production articles
are inserted. Cases are explicitly synthetic, not publishable news.
"""
import asyncio,json,math
from pathlib import Path
from types import SimpleNamespace
from langfuse import Langfuse
from langchain_openai import ChatOpenAI
from agent_core.config import Settings
from agent_core.dedup import Embeddings
from custom_agents.news import NewsWorkflow

settings=Settings()
lf=Langfuse()
name='news-novelty-regression'
lf.create_dataset(name=name,description='Synthetic cross-language duplicate, new-development and unrelated controls. Versioned in Git.')
for i,case in enumerate(json.loads((Path(__file__).resolve().parents[1]/'evals/news-novelty.json').read_text())):
    lf.create_dataset_item(dataset_name=name,id='news-novelty-v1-'+str(i),**case)

async def task(*,item,**kwargs):
    data=item.input if hasattr(item,'input') else item['input']
    w=object.__new__(NewsWorkflow)
    w.embeddings=Embeddings(settings)
    w.model=ChatOpenAI(model=settings.text_model,base_url=settings.litellm_url,api_key=settings.litellm_key,
        temperature=0,max_tokens=1200,timeout=300,extra_body={'chat_template_kwargs':{'enable_thinking':False}})
    a,b=await w.embeddings.embed([data['proposed'],data['previous']])
    similarity=sum(x*y for x,y in zip(a,b))
    async def nearest(*args,**kwargs):
        return [{'job_id':'00000000-0000-0000-0000-000000000001','title':'Synthetic archive article',
                 'content':data['previous'],'similarity':similarity}]
    w.archive=SimpleNamespace(nearest=nearest)
    return await w.check_novelty({'job_id':'00000000-0000-0000-0000-000000000002','brief':data['proposed']},data['proposed'])

def accuracy(*,output,expected_output,**kwargs):
    return {'name':'novelty_accuracy','value':int(output['duplicate']==expected_output),'data_type':'BOOLEAN',
            'comment':output['reason']}

result=lf.run_experiment(name='BGE-M3 Q8 + Qwen novelty',data=lf.get_dataset(name).items,
    task=task,evaluators=[accuracy],max_concurrency=1,metadata={'suite':'news-novelty-v1','corpus':'synthetic-controls'})
lf.flush()
print('Experiment:',result.experiment_url)
for r in result.item_results:print(r.output,r.evaluations)
if any(not r.output or not r.evaluations or r.evaluations[0].value!=1 for r in result.item_results):
    raise SystemExit('Novelty regression failed')
