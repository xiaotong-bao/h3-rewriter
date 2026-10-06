"""Compare EP3 and GRPO with exactly the SFT 101-case generation protocol."""
import argparse,json,os,pathlib,re
import torch
import torch.distributed as dist
from peft import PeftModel
from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration
from train_grpo import call
from format_rules import check_format

from run_config import JOB as ROOT, PORT, JUDGE_REVISION
parser=argparse.ArgumentParser()
parser.add_argument('--checkpoint',type=pathlib.Path,default=ROOT/'pilot/final_adapter')
parser.add_argument('--output',type=pathlib.Path,default=ROOT/'evaluation_101')
args=parser.parse_args()
assert (args.checkpoint/'adapter_model.safetensors').is_file(), 'Checkpoint adapter is missing'
SFT=pathlib.Path('/work/trl_sft_official_v2_20261006')
rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
torch.cuda.set_device(local)
if world>1:dist.init_process_group('nccl')
from prepare import user_content
source=pathlib.Path('/work/benchmark_jobs/arena_original_101_retention_v2_20261005/data/inputs.jsonl')
inputs=[json.loads(s) for s in source.read_text().splitlines()]
baseline={r['id']:r for r in map(json.loads,(SFT/'epoch_benchmarks/step906/results.jsonl').read_text().splitlines())}
assert len(inputs)==len(baseline)==101
p=AutoProcessor.from_pretrained(ROOT/'sft_init')
p.image_processor.size={'shortest_edge':3136,'longest_edge':200704}
model=Qwen3_5ForConditionalGeneration.from_pretrained(ROOT/'sft_init',dtype=torch.bfloat16,attn_implementation='sdpa').cuda()
model=PeftModel.from_pretrained(model,args.checkpoint).eval()
out=args.output;out.mkdir(parents=True,exist_ok=True);path=out/f'rank{rank}.jsonl'
done={r['id'] for r in map(json.loads,path.read_text().splitlines())} if path.exists() else set()
for row in inputs[rank::world]:
 if row['id'] in done:continue
 original=row['conversations'][0]['value']
 messages=[{'role':'system','content':[{'type':'text','text':row['system']}]},
  {'role':'user','content':user_content(original,row.get('images',[]),row.get('videos',[]))}]
 assert original.replace('<image>','').strip()==baseline[row['id']]['original_prompt'].strip()
 tokens=p.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=True,return_tensors='pt').to(model.device)
 # Matched zero-GRPO control isolates GRPO from BF16 SFT-adapter merge rounding.
 with model.disable_adapter(),torch.inference_mode():initial_ids=model.generate(**tokens,do_sample=False,max_new_tokens=2048,repetition_penalty=1.,eos_token_id=p.tokenizer.eos_token_id,pad_token_id=p.tokenizer.pad_token_id,use_cache=True)
 initial_rewrite=p.decode(initial_ids[0,tokens['input_ids'].shape[1]:],skip_special_tokens=True).strip()
 with torch.inference_mode():generated=model.generate(**tokens,do_sample=False,max_new_tokens=2048,repetition_penalty=1.,eos_token_id=p.tokenizer.eos_token_id,pad_token_id=p.tokenizer.pad_token_id,use_cache=True)
 ids=generated[0,tokens['input_ids'].shape[1]:];rewrite=p.decode(ids,skip_special_tokens=True).strip()
 context='Requested task:'+row['system'].split('Requested task:',1)[1]
 source_text=baseline[row['id']]['original_prompt']
 old=call({'original':source_text,'context':context,'rewrite':baseline[row['id']]['rewrite']})
 initial=call({'original':source_text,'context':context,'rewrite':initial_rewrite})
 new=call({'original':source_text,'context':context,'rewrite':rewrite})
 task=re.search(r'Requested task:\s*(\w+)',context)[1]
 duration=float(re.search(r'duration:\s*([0-9.]+)s',context)[1])
 old_format=check_format(baseline[row['id']]['rewrite'],task,duration);new_format=check_format(rewrite,task,duration)
 initial_format=check_format(initial_rewrite,task,duration)
 result={'initial_rewrite':initial_rewrite,'initial_audit':initial,'initial_format':initial_format,'baseline_format':old_format,'grpo_format':new_format,'id':row['id'],'original_prompt':source_text,'baseline_rewrite':baseline[row['id']]['rewrite'],
  'grpo_rewrite':rewrite,'generated_tokens':len(ids),'at_token_cap':len(ids)>=2048,'baseline_audit':old,'grpo_audit':new}
 with path.open('a') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n')
 print('EVALUATED',row['id'],old['reward'],new['reward'],flush=True)
if world>1:dist.barrier()
if rank==0:
 rows=[json.loads(s) for i in range(world) for s in (out/f'rank{i}.jsonl').read_text().splitlines()]
 assert len(rows)==len({r['id'] for r in rows})==101
 valid=[r for r in rows if not r['baseline_audit'].get('judge_failed') and not r['grpo_audit'].get('judge_failed') and not r['initial_audit'].get('judge_failed')]
 summary={'checkpoint':str(args.checkpoint),'judge_revision':JUDGE_REVISION,'rows':101,'valid_pairs':len(valid),'judge_failed_pairs':101-len(valid),
  'scope':'GPT-6 Luna exploratory paired audit; independent Codex session case review still required'}
 for label in ('baseline','initial','grpo'):
  values=[r[label+'_audit'] for r in valid]
  summary[label]={'format_failures':sum(not all(r[label+'_format'].values()) for r in rows),'severe_cases':sum(v.get('severity_counts',{}).get('severe',0)>0 for v in values),'general_cases':sum(v.get('severity_counts',{}).get('general',0)>0 for v in values),'clear_cases':sum(any(i['status'] in ('omitted','contradicted') for i in v['items']) for v in values),
   'mean_reward':sum(v['reward'] for v in values)/len(values) if values else None}
 (out/'summary.json').write_text(json.dumps(summary,indent=2))
 (out/'comparison.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in sorted(rows,key=lambda r:r['id'])))
 (out/'COMPLETE').write_text('101\n')
if world>1:dist.destroy_process_group()
