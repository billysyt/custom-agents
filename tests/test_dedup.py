import uuid
from datetime import datetime,timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
from agent_core.dedup import fingerprint,overlap_score,vector_literal,chunks
from custom_agents.news import NewsWorkflow
from custom_agents.a2a import task


def test_copy_detection_ignores_citations_and_formatting():
    source='這是一段具有足夠長度的原始新聞報道，描述一項模型發布與授權條款。'*10
    assert overlap_score(source+'[1]',source)==1
    assert fingerprint('News [1] ABC')==fingerprint('news abc')
    assert overlap_score(source,'完全不同的天氣消息。'*50)==0
    assert all(len(c)<=700 for c in chunks(source*10))
    with pytest.raises(ValueError):vector_literal([float('nan')]*1024)


async def test_novelty_uses_new_facts_judge_not_cosine_alone():
    w=object.__new__(NewsWorkflow)
    w.embeddings=SimpleNamespace(embed=AsyncMock(return_value=[[1.0]*1024]))
    previous=str(uuid.uuid4())
    w.archive=SimpleNamespace(nearest=AsyncMock(return_value=[{'job_id':previous,'title':'Old release','content':'old facts','similarity':0.94}]))
    w.judge=AsyncMock(return_value={'duplicate':False,'matched_job_id':previous,'new_facts':['new security fix'],'reason':'A subsequent verified security update.'})
    state={'job_id':str(uuid.uuid4()),'brief':'new facts'}
    assert not (await w.check_novelty(state,'update'))['duplicate']
    w.judge.return_value={'duplicate':True,'matched_job_id':previous,'new_facts':[],'reason':'Same event in another language.'}
    assert (await w.check_novelty(state,'translation'))['duplicate']
    w.judge.return_value['matched_job_id']=str(uuid.uuid4())
    with pytest.raises(ValueError):await w.check_novelty(state,'translation')


def test_duplicate_a2a_is_completed_without_article_or_cover():
    row={'id':uuid.uuid4(),'status':'completed','stage':'duplicate','updated_at':datetime.now(timezone.utc),
         'error':None,'result':{'status':'duplicate','reason':'Same verified event already covered.'}}
    result=task(row,SimpleNamespace(public_url='https://example.com'))
    assert result['status']['state']=='completed'
    assert result['artifacts'][0]['artifactId']=='duplicate'
