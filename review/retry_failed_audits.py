"""Retry failed judges on saved raw rewrites, preserving the original evaluation."""
import argparse,concurrent.futures,json,pathlib,time,urllib.request
p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--port',type=int,default=8792);a=p.parse_args()
job=a.root/'grpo_ep3_luna_v2_20261006';src=job/'evaluation_101';out=src/'recovered';out.mkdir(exist_ok=True)
assert (src/'COMPLETE').is_file(),'Wait for generation and initial audit to finish'
inputs={r['id']:r for r in map(json.loads,(a.root/'benchmark_jobs/arena_original_101_retention_v2_20261005/data/inputs.jsonl').read_text().splitlines())}
rows=[json.loads(s) for s in (src/'comparison.jsonl').read_text().splitlines()]
assert len(rows)==len({r['id'] for r in rows})==101
cache=out/'recovered_verdicts.jsonl';done={}
if cache.exists():
 for line in cache.read_text().splitlines():
  r=json.loads(line)
  if not r['verdict'].get('judge_failed'):done[(r['id'],r['label'])]=r['verdict']
tasks=[]
for r in rows:
 for label in ('baseline','initial','grpo'):
  if r[label+'_audit'].get('judge_failed'):
   key=(r['id'],label)
   if key in done:r[label+'_audit']=done[key]
   else:tasks.append((r,label))
def judge(task):
 r,label=task;context='Requested task:'+inputs[r['id']]['system'].split('Requested task:',1)[1]
 payload={'original':r['original_prompt'],'context':context,'rewrite':r[label+'_rewrite']}
 errors=[]
 for attempt in range(3):
  try:
   req=urllib.request.Request(f'http://127.0.0.1:{a.port}/score',json.dumps(payload).encode(),{'Content-Type':'application/json'})
   with urllib.request.urlopen(req,timeout=650) as response:v=json.load(response)
   assert isinstance(v.get('reward'),(int,float))
   return r,label,v
  except Exception as error:errors.append(repr(error));time.sleep(2)
 return r,label,{'judge_failed':True,'errors':errors}
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
 for r,label,v in pool.map(judge,tasks):
  with cache.open('a') as f:f.write(json.dumps({'id':r['id'],'label':label,'verdict':v},ensure_ascii=False)+'\n')
  r[label+'_audit']=v
valid=[r for r in rows if all(not r[x+'_audit'].get('judge_failed') for x in ('baseline','initial','grpo'))]
summary={'rows':101,'valid_pairs':len(valid),'judge_failed_pairs':101-len(valid),'retried_verdicts':len(tasks),'scope':'Luna exploratory paired audit; original raw rewrites unchanged; original audit files retained'}
for label in ('baseline','initial','grpo'):
 values=[r[label+'_audit'] for r in valid]
 summary[label]={'format_failures':sum(not all(r[label+'_format'].values()) for r in rows),'severe_cases':sum(v.get('severity_counts',{}).get('severe',0)>0 for v in values),'general_cases':sum(v.get('severity_counts',{}).get('general',0)>0 for v in values),'clear_cases':sum(any(i['status'] in ('omitted','contradicted') for i in v.get('items',[])) for v in values),'mean_reward':sum(v['reward'] for v in values)/len(values) if values else None}
(out/'comparison.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
