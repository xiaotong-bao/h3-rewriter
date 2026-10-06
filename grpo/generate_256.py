import json,os,pathlib,torch
from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration
from peft import PeftModel
ROOT=pathlib.Path('/work/grpo_ep3_luna_v2_20261006');rank=int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(rank)
p=AutoProcessor.from_pretrained('/work/models/Qwen3.5-9B');p.image_processor.size={'shortest_edge':3136,'longest_edge':200704}
m=Qwen3_5ForConditionalGeneration.from_pretrained('/work/models/Qwen3.5-9B',dtype=torch.bfloat16,attn_implementation='sdpa').cuda()
m=PeftModel.from_pretrained(m,'/work/trl_sft_official_v2_20261006/run/checkpoint-906').eval()
rows=[json.loads(s) for s in (ROOT/'pilot_inputs.jsonl').read_text().splitlines()]
path=ROOT/f'ep3_samples_rank{rank}.jsonl';done={r['id'] for r in map(json.loads,path.read_text().splitlines())} if path.exists() else set()
for r in rows[rank::8]:
 if r['id'] in done:continue
 tokens=p.apply_chat_template(r['prompt'],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=True,return_tensors='pt').to(m.device)
 with torch.inference_mode():g=m.generate(**tokens,do_sample=False,max_new_tokens=2048,repetition_penalty=1.,eos_token_id=p.tokenizer.eos_token_id,pad_token_id=p.tokenizer.pad_token_id,use_cache=True)
 ids=g[0,tokens['input_ids'].shape[1]:];r.update(rewrite=p.decode(ids,skip_special_tokens=True).strip(),generated_tokens=len(ids),at_token_cap=len(ids)>=2048)
 with path.open('a') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 print(rank,r['id'],len(ids),flush=True)
