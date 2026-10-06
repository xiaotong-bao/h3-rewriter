"""Persist requirement-based questions and a mandatory whole-source final-state check."""
import argparse,concurrent.futures,json,pathlib,time,urllib.request
from reward_policy import validate_requirements
from evidence import numbered_spans
from run_config import JUDGE_REVISION,PORT

def design(row,port):
 req=urllib.request.Request(f'http://127.0.0.1:{port}/requirements',json.dumps({'original':row['original'],'context':row['context']}).encode(),{'Content-Type':'application/json'})
 for attempt in range(3):
  try:
   with urllib.request.urlopen(req,timeout=1800) as response:requirements=json.load(response)['requirements']
   validate_requirements(row['original'],requirements);break
  except Exception:
   if attempt==2:raise
   time.sleep(5*(attempt+1))
 return {'id':row['id'],'original':row['original'],'context':row['context'],'judge_revision':JUDGE_REVISION,
         'requirements':requirements,
         'questions':[{'requirement_id':r['id'],'critical':r['critical'],'source_ids':r['source_ids'],
                       'question':'Does the ENTIRE rewrite preserve this explicit requirement, with the same entities, binding, direction, quantity and timing? '+r['text']} for r in requirements],
         'state_questions':[{'source_id':s['id'],'source_text':s['text'],
                             'question':'What initial state, change, and final state does this source sentence require? Read every later statement about the same objects. Does the LAST described state contradict that result? Cite supporting and conflicting spans; do not rely on an earlier matching verb.'} for s in numbered_spans(row['original'])],
         'validation':'Structural coverage validated; semantic correctness is not certified.'}

def main():
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--port',type=int,default=PORT);p.add_argument('--workers',type=int,default=16);a=p.parse_args();assert 1<=a.workers<=16
 with urllib.request.urlopen(f'http://127.0.0.1:{a.port}/',timeout=5) as response:health=json.load(response)
 assert health['revision']==JUDGE_REVISION and health['runtime']=='local_codex'
 rows=[json.loads(l) for l in a.inputs.read_text().splitlines()]
 existing=[json.loads(l) for l in a.output.read_text().splitlines()] if a.output.exists() else []
 lookup={r['id']:r for r in rows}
 for r in existing:
  assert r['judge_revision']==JUDGE_REVISION and r['original']==lookup[r['id']]['original'] and r['context']==lookup[r['id']]['context']
 done={r['id'] for r in existing};pending=[r for r in rows if r['id'] not in done];errors=[]
 started=time.monotonic()
 with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool,a.output.open('a') as out:
  futures={pool.submit(design,r,a.port):r for r in pending}
  for future in concurrent.futures.as_completed(futures):
   row=futures[future]
   try:
    result=future.result();out.write(json.dumps(result,ensure_ascii=False)+'\n');out.flush();done.add(row['id'])
    print(f'QUESTIONS_READY {len(done)}/{len(rows)}',flush=True)
   except Exception as e:errors.append({'id':row['id'],'error':repr(e)});print('FAILED',row['id'],repr(e),flush=True)
 report={'passed':len(done)==len(rows) and not errors,'ready':len(done),'total':len(rows),'errors':errors,'seconds':time.monotonic()-started,'judge_revision':JUDGE_REVISION}
 a.output.with_suffix('.receipt.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
 if not report['passed']:raise SystemExit(1)
if __name__=='__main__':main()
