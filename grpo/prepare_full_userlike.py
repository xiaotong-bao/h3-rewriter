"""Prepare all userlike tasks, excluding held-out text and media contents."""
import collections,hashlib,json,re
from pathlib import Path
ROOT=Path('/data/xiaotong/h3_rewriter_sft_20261002')
JOB=ROOT/'grpo_9bv2_userlike_full_resume_20261008'
def read(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def original(r):return re.sub(r'<(?:image|video)>','',r['conversations'][0]['value']).strip()
def digest(s):return hashlib.sha256(s.encode()).hexdigest()
def main():
 JOB.mkdir(exist_ok=True)
 train=read(ROOT/'userlike_gemini_20261007/userlike/train.jsonl')
 held=read(ROOT/'userlike_gemini_20261007/userlike/val.jsonl')+read(ROOT/'lf_dataset/val.jsonl')+read(ROOT/'benchmark_jobs/arena_original_101_retention_v2_20261005/data/inputs.jsonl')
 blocked={digest(original(r)) for r in held};cache={}
 def media_sha(path):
  if path not in cache:
   local=ROOT/Path(path).relative_to('/work');h=hashlib.sha256()
   with local.open('rb') as f:
    for b in iter(lambda:f.read(8388608),b''):h.update(b)
   cache[path]=h.hexdigest()
  return cache[path]
 blocked_media={media_sha(m) for r in held for m in r.get('images',[])+r.get('videos',[])}
 rows=[];excluded=collections.Counter()
 for r in train:
  text=original(r)
  if digest(text) in blocked:excluded['heldout_text']+=1;continue
  if any(media_sha(m) in blocked_media for m in r.get('images',[])+r.get('videos',[])):excluded['heldout_media']+=1;continue
  task=re.search(r'Requested task:\s*(\w+)',r['system'])[1]
  media={'image':iter(r.get('images',[])),'video':iter(r.get('videos',[]))};content=[]
  for part in re.split(r'(<image>|<video>)',r['conversations'][0]['value']):
   if part in ('<image>','<video>'):
    kind=part[1:-1];content.append({'type':kind,kind:next(media[kind])})
   elif part:content.append({'type':'text','text':part})
  assert all(next(v,None) is None for v in media.values())
  rows.append(dict(id=r['id'],original=text,context='Requested task:'+r['system'].split('Requested task:',1)[1],task=task,prompt=[{'role':'system','content':[{'type':'text','text':r['system']}]},{'role':'user','content':content}]))
 target=JOB/'pilot_inputs.jsonl';assert not target.exists();target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
 report=dict(source_rows=len(train),selected=len(rows),tasks=dict(collections.Counter(r['task'] for r in rows)),excluded=dict(excluded),input_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),checkpoint='/work/grpo_9bv2_userlike_1k_20261007/pilot/checkpoint-500',extra_steps=(len(rows)+1)//2)
 (JOB/'data_report.json').write_text(json.dumps(report,indent=2)+'\n');(JOB/'single_pass_authorization.json').write_text(json.dumps(dict(approved=True,revision='retention-single-pass-v3-luna-userlike-v3-astra-guard',scope='User-requested full userlike strict resume from step500; updated learning rate; every100 Astra101; startup calibration',source_sha256=sorted({digest(r['original'].strip()) for r in rows})),indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
