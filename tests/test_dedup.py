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
    w.archive=SimpleNamespace(nearest=AsyncMock(return_value=[{'job_id':previous,'title':'Old release','content':'ExampleLab launched Model A. It supports a 32k context window.','similarity':0.94}]))
    w.judge=AsyncMock(return_value={'duplicate':False,'matched_job_id':previous,'new_facts':['new security fix'],'reason':'A subsequent verified security update.'})
    state={'job_id':str(uuid.uuid4()),'brief':'new facts'}
    assert not (await w.check_novelty(state,'update'))['duplicate']
    w.judge.return_value={'duplicate':True,'matched_job_id':previous,'new_facts':[],'reason':'Same event in another language.',
        'matching_facts':[{'proposed_quote':'ExampleLab launched Model A','previous_quote':'ExampleLab launched Model A'},
                          {'proposed_quote':'It supports a 32k context window','previous_quote':'It supports a 32k context window'}]}
    assert (await w.check_novelty(state,'ExampleLab launched Model A. It supports a 32k context window.'))['duplicate']
    # A model explanation that invents evidence in the archive must fail closed.
    with pytest.raises(ValueError):await w.check_novelty(state,'A completely different news event.')
    w.judge.return_value['matched_job_id']=str(uuid.uuid4())
    with pytest.raises(ValueError):await w.check_novelty(state,'translation')


def test_duplicate_a2a_is_completed_without_article_or_cover():
    row={'id':uuid.uuid4(),'status':'completed','stage':'duplicate','updated_at':datetime.now(timezone.utc),
         'error':None,'result':{'status':'duplicate','reason':'Same verified event already covered.'}}
    result=task(row,SimpleNamespace(public_url='https://example.com'))
    assert result['status']['state']=='completed'
    assert result['artifacts'][0]['artifactId']=='duplicate'


async def test_duplicate_evidence_repair_is_bounded_and_still_checks_quotes():
    w=object.__new__(NewsWorkflow)
    w.embeddings=SimpleNamespace(embed=AsyncMock(return_value=[[1.0]*1024]))
    previous=str(uuid.uuid4())
    text='ExampleLab launched Model A. It supports a 32k context window.'
    w.archive=SimpleNamespace(nearest=AsyncMock(return_value=[{
        'job_id':previous,'title':'Old release','content':text,'similarity':0.94}]))
    valid={'duplicate':True,'matched_job_id':previous,'new_facts':[],'reason':'Same model release.',
        'matching_facts':[{'proposed_quote':'ExampleLab launched Model A','previous_quote':'ExampleLab launched Model A'},
                          {'proposed_quote':'It supports a 32k context window','previous_quote':'It supports a 32k context window'}]}
    invalid={**valid,'matching_facts':[{'proposed_quote':'ExampleLab launched...32k context window',
        'previous_quote':'ExampleLab launched...32k context window'}]*2}
    state={'job_id':str(uuid.uuid4())}
    w.judge=AsyncMock(side_effect=[invalid,valid])
    assert (await w.check_novelty(state,text))['duplicate']
    assert w.judge.await_count==2
    assert 'validation_feedback' in w.judge.call_args.args[2]
    w.judge=AsyncMock(return_value=invalid)
    with pytest.raises(ValueError,match='verifiable matching facts'):
        await w.check_novelty(state,text)
    assert w.judge.await_count==2
