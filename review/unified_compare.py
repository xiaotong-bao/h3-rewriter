"""Blind, same-protocol audit of saved FAL/EP3/merged/GRPO outputs; no regeneration."""
import argparse,concurrent.futures,hashlib,json,pathlib,re,time,urllib.request,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'grpo'))
from format_rules import check_format
from reward_policy import validate_and_score
LABELS=('fal','ep3','merged_ep3','step32','step64')
def read(p):return {r['id']:r for r in map(json.loads,p.read_text().splitlines())}
def summarize(rows,out,revision):
 table={};common=[]
 byid={}
 for r in rows:byid.setdefault(r['id'],{})[r['label']]=r
 for key,group in byid.items():
  if set(group)==set(LABELS) and all(not r['audit'].get('judge_failed') for r in group.values()):common.append(key)
 for label in LABELS:
  allrows=[r for r in rows if r['label']==label];v=[r for r in allrows if r['id'] in common];audits=[r['audit'] for r in v]
  table[label]={'completed':len(allrows),'valid_common':len(v),'judge_failed':sum(r['audit'].get('judge_failed',False) for r in allrows),
   'severe':sum(a['severity_counts']['severe']>0 for a in audits),'highest_general':sum(a['maximum_severity']=='general' for a in audits),
   'highest_review':sum(a['maximum_severity']=='review' for a in audits),'none':sum(a['maximum_severity']=='none' for a in audits),
   'music':sum(a['unrequested_music'] for a in audits),'dialogue':sum(a['unrequested_dialogue'] for a in audits),
   'format_failures':sum(not all(r['format'].values()) for r in allrows),
   'mean_reward':sum(a['reward'] for a in audits)/len(v) if v else None}
 summary={'revision':revision,'model':'gpt-6-astra','common_valid':len(common),'expected_cases':101,'complete':len(rows)==505 and len(common)==101,
          'scope':'Automated blinded text audit; not independent human review or generated-video evaluation. Smiley case was used in reward development.',
          'models':table}
 (out/'summary.json').write_text(json.dumps(summary,indent=2))
 lines=['# FAL 与当前 EP3/GRPO：统一评分对照','',summary['scope'],'',f'评分：GPT-6 Astra high / {revision}；共同有效配对 {len(common)}/101。',
        '','| 模型 | 已审 | 严重 | 最高一般 | 待复核 | 无问题 | raw格式失败 | 新增配乐 | 新增台词 | 平均reward |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for label,v in table.items():
  reward='—' if v['mean_reward'] is None else f"{v['mean_reward']:.4f}"
  lines.append(f"| {label} | {v['completed']}/101 | {v['severe']} | {v['highest_general']} | {v['highest_review']} | {v['none']} | {v['format_failures']} | {v['music']} | {v['dialogue']} | {reward} |")
 lines+=['','语义统计仅用五组共同有效配对；格式分母为各组已审 raw 输出。不得将部分集当作完整 101 结论。',
         '机械格式使用同一检查器，接受 Image 1 与 <Picture 1>；不清洗输出、不重采样。图像忠实度、H3 encoder gate 和实际视频效果未验证。']
 (out/'COMPARISON.md').write_text('\n'.join(lines)+'\n');return summary

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--port',type=int,default=8796);p.add_argument('--workers',type=int,default=8);a=p.parse_args();assert 1<=a.workers<=16
 b=a.root;fal=read(b/'ep2_dimensions_fal_base_101_20261006/fal_inputs.jsonl');g=read(b/'grpo_ep3_luna_v4_20261006/evaluation_101/comparison.jsonl');s=read(b/'grpo_ep3_luna_v4_20261006/evaluation_101_step32/comparison.jsonl');inputs=read(b/'benchmark_jobs/arena_original_101_retention_v2_20261005/data/inputs.jsonl')
 assert set(fal)==set(g)==set(s)==set(inputs) and len(g)==101
 with urllib.request.urlopen(f'http://127.0.0.1:{a.port}/',timeout=5) as response:health=json.load(response)
 assert health['model']=='gpt-6-astra' and health['runtime']=='local_codex';revision=health['revision']
 tasks=[]
 for key in sorted(g):
  r=g[key];assert fal[key]['original_prompt'].strip()==r['original_prompt'].strip()==s[key]['original_prompt'].strip()
  context='Requested task:'+inputs[key]['system'].split('Requested task:',1)[1]
  for label,text in [('fal',fal[key]['rewrite']),('ep3',r['baseline_rewrite']),('merged_ep3',r['initial_rewrite']),('step32',s[key]['grpo_rewrite']),('step64',r['grpo_rewrite'])]:
   tasks.append({'id':key,'label':label,'original':r['original_prompt'],'context':context,'rewrite':text,'rewrite_sha256':hashlib.sha256(text.encode()).hexdigest()})
 a.output.mkdir(parents=True,exist_ok=True);path=a.output/'audits.jsonl';existing=[json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
 for r in existing:assert r['revision']==revision
 done={(r['id'],r['label']):r for r in existing};manifest={(r['id'],r['label']):r for r in tasks}
 for k,r in done.items():assert r['rewrite_sha256']==manifest[k]['rewrite_sha256']
 (a.output/'manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in tasks))
 def judge(row):
  req=urllib.request.Request(f'http://127.0.0.1:{a.port}/score',json.dumps({k:row[k] for k in ['original','context','rewrite']}).encode(),{'Content-Type':'application/json'})
  errors=[]
  for attempt in range(3):
   try:
    with urllib.request.urlopen(req,timeout=1800) as response:v=json.load(response)
    assert v['judge_model']=='gpt-6-astra' and v['judge_revision']==revision and v['reasoning_effort']=='high'
    reported=v['reward'];validate_and_score(row['original'],row['rewrite'],v['requirements'],v,require_state_audit=True);assert abs(v['reward']-reported)<1e-8
    break
   except Exception as e:errors.append(repr(e));time.sleep(2*(attempt+1))
  else:v={'judge_failed':True,'errors':errors}
  task=re.search(r'Requested task:\s*(\w+)',row['context'])[1];duration=float(re.search(r'duration:\s*([0-9.]+)s',row['context'])[1])
  return {**row,'revision':revision,'audit':v,'format':check_format(row['rewrite'],task,duration)}
 pending=[r for r in tasks if (r['id'],r['label']) not in done or done[(r['id'],r['label'])]['audit'].get('judge_failed')]
 with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool,path.open('a') as out:
  futures={pool.submit(judge,r):r for r in pending}
  for future in concurrent.futures.as_completed(futures):
   r=future.result();done[(r['id'],r['label'])]=r;out.write(json.dumps(r,ensure_ascii=False)+'\n');out.flush()
   print('AUDITED',len(done),'/505',r['id'],r['label'],r['audit'].get('reward'),flush=True)
   summarize(list(done.values()),a.output,revision)
 summary=summarize(list(done.values()),a.output,revision)
 if summary['complete']:(a.output/'COMPLETE').write_text('101 x 5 valid\n')
 else:raise SystemExit('Incomplete audit; resume to retry failed judgments')
if __name__=='__main__':main()
