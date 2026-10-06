import collections,hashlib,json,pathlib,random,re
from run_config import JOB
import torch
from peft import PeftModel
from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration

ROOT=pathlib.Path('/work')
SFT=ROOT/'trl_sft_official_v2_20261006';BASE=ROOT/'models/Qwen3.5-9B'
JOB.mkdir(exist_ok=True)
def read(path):return [json.loads(s) for s in path.read_text().splitlines()]
def digest(text):return hashlib.sha256(text.strip().encode()).hexdigest()
def original(row):return ''.join(x.get('text','') for x in json.loads(row['prompt_json'])[1]['content']).strip()
train=read(SFT/'train.jsonl');val=read(SFT/'val.jsonl')
bench=read(SFT/'epoch_benchmarks/step906/results.jsonl')
blocked={digest(original(r)) for r in val}|{digest(r['original_prompt']) for r in bench}
heldout_media={v for r in val for v in r['images']+r['videos']}
groups=collections.defaultdict(list)
for row in train:
    if row['videos'] or digest(original(row)) in blocked or set(row['images'])&heldout_media:continue
    messages=json.loads(row['prompt_json']);system=messages[0]['content'][0]['text']
    context='Requested task:'+system.split('Requested task:',1)[1]
    task=re.search(r'Requested task:\s*(\w+)',context)[1]
    if task not in ('t2va','i2va'):continue
    groups[task].append({'id':row['id'],'original':original(row),'context':context,'task':task,'prompt':messages})
rows=[]
for task in ('t2va','i2va'):
    random.Random(906).shuffle(groups[task]);rows+=groups[task][:128]
random.Random(906).shuffle(rows)
assert len(rows)==256 and len({r['id'] for r in rows})==256
(JOB/'pilot_inputs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
approved=[digest(r['original']) for r in rows]
(JOB/'luna_approved_sources.json').write_text(json.dumps({'sha256':approved}))
(JOB/'system_prompt.txt').write_bytes((SFT/'system_prompt.txt').read_bytes())
(JOB/'data_report.json').write_text(json.dumps({'selected':len(rows),'tasks':dict(collections.Counter(r['task'] for r in rows)),
    'source_train':str(SFT/'train.jsonl'),'excluded_101_and_validation':True,'seed':906,
    'media_order':'identical prompt_json from official-template SFT','video_scope':'text and image initial pilot only'},indent=2))
checkpoint=SFT/'run/checkpoint-906'
state=json.loads((checkpoint/'trainer_state.json').read_text())
assert state['global_step']==906 and abs(state['epoch']-3)<1e-6
output=JOB/'sft_init';assert not output.exists(),'Never overwrite initialization'
p=AutoProcessor.from_pretrained(BASE)
base=Qwen3_5ForConditionalGeneration.from_pretrained(BASE,dtype=torch.bfloat16,attn_implementation='sdpa',device_map='cuda:0')
model=PeftModel.from_pretrained(base,checkpoint).eval()
# Compare logits before/after BF16 merge on actual official inputs.
messages=next(r['prompt'] for r in rows if r['task']=='t2va')
tokens=p.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=True,return_tensors='pt').to(model.device)
with torch.inference_mode():before=model(**tokens).logits[:,-1].float().cpu()
model=model.merge_and_unload().eval()
with torch.inference_mode():after=model(**tokens).logits[:,-1].float().cpu()
assert before.argmax(-1).item()==after.argmax(-1).item(),'Merge changed next-token argmax'
model.generation_config.eos_token_id=p.tokenizer.eos_token_id
model.generation_config.pad_token_id=p.tokenizer.pad_token_id
model.save_pretrained(output,safe_serialization=True,max_shard_size='5GB');p.save_pretrained(output)
assert (output/'chat_template.jinja').read_bytes()==(BASE/'chat_template.jinja').read_bytes()
receipt={'checkpoint':str(checkpoint),'epoch':3,'step':906,
    'adapter_sha256':hashlib.sha256((checkpoint/'adapter_model.safetensors').read_bytes()).hexdigest(),
    'template_sha256':hashlib.sha256((output/'chat_template.jinja').read_bytes()).hexdigest(),
    'merge_next_token_argmax_equal':True,'merge_max_logit_difference':(before-after).abs().max().item(),
    'eos_token_id':p.tokenizer.eos_token_id,'reference_model':'merged EP3; GRPO adapter disabled yields EP3'}
(JOB/'initialization_receipt.json').write_text(json.dumps(receipt,indent=2))
(JOB/'MERGE_READY').write_text('EP3 checkpoint-906 merged and validated\n')
print(json.dumps(receipt),flush=True)
