const base='/agents/news';
let selected=new URLSearchParams(location.search).get('job');
let submitting=false;
const el=id=>document.getElementById(id);
const names={queued:'等候處理',running:'處理中',failed:'需要重試',completed:'草稿完成'};
const stages=[['research','研究來源'],['draft','撰寫草稿'],['review','核實編輯'],['cover','生成封面'],['save','保存稿件']];
async function api(path,options={}){
  const r=await fetch(base+path,{...options,headers:{'Content-Type':'application/json',...options.headers}});
  if(!r.ok){let e=await r.json().catch(()=>({}));throw Error(typeof e.detail==='string'?e.detail:(e.error||'服務暫時未能回應'));}
  return r.json();
}
function select(id){selected=id;history.replaceState(null,'','?job='+id);refresh().catch(showError);}
function showError(e){el('form-error').textContent=e.message;}
function renderJob(j){
  el('empty').hidden=true;el('job-view').hidden=false;
  el('status').textContent=names[j.status];el('status').className='pill '+j.status;
  el('created').textContent=new Date(j.created_at).toLocaleString('zh-HK');
  const current=stages.findIndex(s=>s[0]===j.stage);
  el('stages').replaceChildren(...stages.filter(s=>j.request.cover||s[0]!=='cover').map(([id,label],i)=>{
    const li=document.createElement('li');li.textContent=label;li.className=j.status==='completed'||i<current?'done':id===j.stage?'active':'';return li;
  }));
  el('job-error').textContent=j.error||'';el('retry').hidden=j.status!=='failed';
  el('progress-note').textContent=j.status==='running'?'正在處理，可以離開此頁；進度會自動保存。':j.status==='queued'?'等候工作程序接手。':'';
  const r=j.result;el('result').hidden=!r;if(!r)return;
  el('title').textContent=r.article.title;el('body').replaceChildren(...r.article.body.split(/\n\s*\n/).map(t=>{const p=document.createElement('p');p.textContent=t;return p;}));
  el('cover-img').parentElement.hidden=!r.cover;if(r.cover)el('cover-img').src=`${base}/jobs/${j.id}/files/cover.png`;
  el('sources').replaceChildren(...r.sources.map(s=>{const li=document.createElement('li'),a=document.createElement('a');a.href=s.url;a.textContent=`[${s.id}] ${s.title}${s.date?' · '+s.date:''}`;a.target='_blank';a.rel='noopener noreferrer';li.append(a);return li;}));
  el('downloads').replaceChildren(...['article.md','result.json',...(r.cover?['cover.png']:[])].map(file=>{const a=document.createElement('a');a.href=`${base}/jobs/${j.id}/files/${file}`;a.textContent=file==='article.md'?'下載文章':file==='cover.png'?'下載封面':'完整資料';a.download=file;return a;}));
  el('metadata').textContent=JSON.stringify({模型:r.text_model,封面模型:r.image_model,處理秒數:r.timings,備註:r.review},null,2);
}
async function refresh(){
  const jobs=await api('/jobs');el('jobs').replaceChildren(...jobs.map(j=>{const b=document.createElement('button');b.className='job-link'+(selected===j.id?' selected':'');const title=document.createElement('strong'),status=document.createElement('span');title.textContent=j.result?.article.title||j.request.topic;status.textContent=names[j.status]+' · '+new Date(j.created_at).toLocaleDateString('zh-HK');b.append(title,status);b.onclick=()=>select(j.id);return b;}));
  if(!selected&&jobs.length)select(jobs[0].id);else if(selected){const j=jobs.find(j=>j.id===selected)||await api('/jobs/'+selected);renderJob(j);}
}
el('new-job').addEventListener('submit',async e=>{e.preventDefault();if(submitting)return;submitting=true;const button=e.target.querySelector('button');button.disabled=true;el('form-error').textContent='';try{const j=await api('/jobs',{method:'POST',headers:{'Idempotency-Key':crypto.randomUUID()},body:JSON.stringify({topic:el('topic').value,language:el('language').value,length:Number(el('length').value),cover:el('cover').checked})});select(j.id);}catch(e){showError(e);}finally{submitting=false;button.disabled=false;}});
el('retry').onclick=async()=>{try{await api('/jobs/'+selected+'/retry',{method:'POST'});el('form-error').textContent='';await refresh();}catch(e){showError(e);}};
refresh().catch(showError);setInterval(()=>{if(!document.hidden)refresh().catch(showError);},5000);
