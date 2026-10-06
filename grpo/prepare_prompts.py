"""Select a reproducible, held-out-isolated GRPO prompt pool without loading GPUs."""
import argparse,collections,hashlib,json,pathlib,random,re

def digest(text):return hashlib.sha256(text.strip().encode()).hexdigest()
def original(row):return ''.join(x.get('text','') for x in json.loads(row['prompt_json'])[1]['content']).strip()
def prepare(sft,output,count=2000):
 def read(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
 train=read(sft/'train.jsonl');val=read(sft/'val.jsonl');bench=read(sft/'epoch_benchmarks/step906/results.jsonl')
 blocked={digest(original(r)) for r in val}|{digest(r['original_prompt']) for r in bench}
 media={v for r in val for v in r['images']+r['videos']}
 groups=collections.defaultdict(list);seen=set()
 for r in train:
  source=original(r);key=digest(source)
  if r['videos'] or key in blocked or key in seen or set(r['images'])&media:continue
  messages=json.loads(r['prompt_json']);system=messages[0]['content'][0]['text']
  context='Requested task:'+system.split('Requested task:',1)[1];task=re.search(r'Requested task:\s*(\w+)',context)[1]
  if task not in ('t2va','i2va'):continue
  seen.add(key);groups[task].append({'id':r['id'],'original':source,'context':context,'task':task,'prompt':messages})
 assert sum(map(len,groups.values()))>=count,'Not enough eligible distinct prompts'
 allocation={t:min(len(groups[t]),count//2) for t in ('t2va','i2va')}
 remaining=count-sum(allocation.values())
 for t in ('t2va','i2va'):
  extra=min(remaining,len(groups[t])-allocation[t]);allocation[t]+=extra;remaining-=extra
 rows=[]
 for task,n in allocation.items():
  assert len(groups[task])>=n,f'Not enough distinct {task} prompts'
  random.Random(906).shuffle(groups[task]);rows+=groups[task][:n]
 random.Random(906).shuffle(rows)
 assert len(rows)==len({digest(r['original']) for r in rows})==count
 output.mkdir(parents=True,exist_ok=True);target=output/'pilot_inputs.jsonl'
 assert not target.exists(),'Never overwrite a frozen prompt pool'
 target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
 report={'selected':count,'tasks':dict(collections.Counter(r['task'] for r in rows)),'seed':906,
         'excluded_validation_and_101_sources':True,'excluded_validation_media':True,'unique_source_hashes':count,
         'source_train':str(sft/'train.jsonl'),'inputs_sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
 (output/'data_report.json').write_text(json.dumps(report,indent=2));return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--sft',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--count',type=int,default=2000);a=p.parse_args();assert a.count>0
 print(json.dumps(prepare(a.sft,a.output,a.count)))
