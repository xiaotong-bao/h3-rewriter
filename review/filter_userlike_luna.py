"""Audit all new userlike training pairs with the frozen Luna reward codebook."""
import argparse,collections,concurrent.futures,hashlib,json,pathlib,re,time,sys
import single_pass_compare as judge

def dump(path,data):
 temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2));temp.replace(path)
def jsonl(path,rows):
 temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows));temp.replace(path)
def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--workers',type=int,default=8);a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
 import fcntl
 lock=(a.output/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 rows=[r for r in map(json.loads,a.source.read_text().splitlines()) if r.get('variant')=='userlike' or r['id'].endswith('__userlike')]
 assert len(rows)==len({r['id'] for r in rows})==9659
 manifest=[]
 for r in rows:
  assert r['conversations'][0]['from']=='human' and r['conversations'][1]['from']=='gpt'
  context=r['system'].split('Requested task:',1)[1];match=re.match(r'\s*([^;]+); aspect ratio: ([^;]+); duration: ([\d.]+)s\.',context);assert match,r['id']
  # Explicit role metadata is source evidence; media content is not sent to the text judge.
  source=r['conversations'][0]['value'].replace('<image>','').replace('<video>','').strip()
  reference=context[match.end():].strip()
  if reference:source+='\n\nSupplied reference roles:\n'+reference
  manifest.append(dict(id=r['id'],label='userlike_teacher',original=source,rewrite=r['conversations'][1]['value'],task=match[1],aspect=match[2],duration_s=float(match[3])))
 receipt=dict(source=str(a.source),source_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),rows=len(rows),model='gpt-6-luna',effort='high',revision=judge.REVISION,rules_sha256=hashlib.sha256(judge.RULES.encode()).hexdigest(),gate='keep if no severe issue and format valid; strict_clean if no issue and format valid',active_training_unchanged=True)
 existing=a.output/'receipt.json'
 if existing.exists():assert json.loads(existing.read_text())==receipt
 else:dump(existing,receipt)
 jsonl(a.output/'manifest.jsonl',manifest)
 audits={};path=a.output/'audits.jsonl'
 if path.exists():
  for line in path.read_text().splitlines():
   if line.strip():
    r=json.loads(line);audits[r['id']]=r
 failed={};initial_valid=len(audits);start=time.time()
 def save_progress():
  counts=collections.Counter(r['maximum_severity'] for r in audits.values());dump(a.output/'status.json',dict(phase='reviewing',total=len(rows),valid=len(audits),failed=len(failed),workers=a.workers,resumed_valid=initial_valid,newly_valid=len(audits)-initial_valid,elapsed_seconds=round(time.time()-start),levels=dict(counts),updated_at=time.time()))
 pending=[r for r in manifest if r['id'] not in audits];save_progress()
 with path.open('a') as out,concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
  futures={pool.submit(judge.grade_retry,r,a.output,'gpt-6-luna','high',judge.REVISION):r for r in pending}
  for f in concurrent.futures.as_completed(futures):
   row=futures[f]
   try:
    result=f.result();audits[row['id']]=result;out.write(json.dumps(result,ensure_ascii=False)+'\n');out.flush()
   except Exception as e:failed[row['id']]=dict(id=row['id'],error=str(e));print('FAILED',row['id'],str(e),flush=True)
   save_progress()
   if (len(audits)+len(failed))%25==0:print('PROGRESS',len(audits),len(failed),'/',len(rows),flush=True)
 lookup={r['id']:r for r in rows};kept=[];strict=[];rejected=[];unscored=[]
 for r in rows:
  verdict=audits.get(r['id'])
  if verdict is None:unscored.append(r);continue
  if verdict['positive_eligible']:kept.append(r)
  else:rejected.append(r)
  if verdict['maximum_severity']=='none' and verdict['format_score']==100:strict.append(r)
 for name,items in [('kept',kept),('strict_clean',strict),('rejected',rejected),('judge_failed',unscored)]:jsonl(a.output/(name+'.jsonl'),items)
 jsonl(a.output/'failures.jsonl',list(failed.values()))
 summary=dict(total=len(rows),valid=len(audits),kept=len(kept),strict_clean=len(strict),rejected=len(rejected),judge_failed=len(unscored),mean_reward=sum(r['reward'] for r in audits.values())/len(audits) if audits else None,levels=dict(collections.Counter(r['maximum_severity'] for r in audits.values())),phase='complete' if not unscored else 'partial',updated_at=time.time())
 dump(a.output/'summary.json',summary);dump(a.output/'status.json',summary)
 if not unscored:(a.output/'COMPLETE').write_text('9659\n')
 print(json.dumps(summary),flush=True)
 if unscored:sys.exit(1)
if __name__=='__main__':main()
