"""Fresh EP3 greedy and sampling inference using the official benchmark inputs."""
import argparse
import copy
import hashlib
import json
import os
import pathlib

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024), b''):h.update(chunk)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser()
    for name in ('base','adapter','inputs','system','output'):
        parser.add_argument('--'+name,type=pathlib.Path,required=True)
    parser.add_argument('--media-from',default='/work/')
    parser.add_argument('--media-root',type=pathlib.Path,required=True)
    parser.add_argument('--seed',type=int,default=20261006)
    parser.add_argument('--modes',nargs='+',choices=['greedy','sampling'],default=['greedy','sampling'])
    a=parser.parse_args()
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    rows=[json.loads(s) for s in a.inputs.read_text().splitlines()]
    assert len(rows)==len({r['id'] for r in rows})==101
    assert a.adapter.name=='checkpoint-906','Require explicitly identified EP3 checkpoint'
    required=['adapter_model.safetensors','adapter_config.json']
    assert all((a.adapter/n).is_file() for n in required)
    assert hashlib.sha256(a.system.read_text().strip().encode()).hexdigest()=='9b95942a5c905bfd8ca703cbb4c31275387e282775395278ff92a889e81d2eda','Unexpected retention v2 system text'
    for r in rows:assert r['system'].startswith(a.system.read_text().strip())
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration,set_seed
    from prepare import user_content
    torch.set_num_threads(1)
    local_rank=int(os.environ.get('LOCAL_RANK',0));torch.cuda.set_device(local_rank)
    device=f'cuda:{local_rank}'
    p=AutoProcessor.from_pretrained(a.base)
    assert p.chat_template==(a.base/'chat_template.jinja').read_text()
    p.image_processor.size={'shortest_edge':3136,'longest_edge':200704}
    p.video_processor.size={'shortest_edge':3136,'longest_edge':50176}
    p.video_processor.fps=1.0;p.video_processor.min_frames=2;p.video_processor.max_frames=8
    p.tokenizer.padding_side='right';assert p.tokenizer.eos_token=='<|im_end|>'
    model=Qwen3_5ForConditionalGeneration.from_pretrained(a.base,dtype=torch.bfloat16,attn_implementation='sdpa').to(device)
    model=PeftModel.from_pretrained(model,a.adapter,is_trainable=False).eval()
    receipt={'base':str(a.base),'adapter':str(a.adapter),'adapter_sha256':sha(a.adapter/'adapter_model.safetensors'),
        'adapter_config_sha256':sha(a.adapter/'adapter_config.json'),'input_sha256':sha(a.inputs),
        'system_sha256':sha(a.system),'template_sha256':hashlib.sha256(p.chat_template.encode()).hexdigest(),
        'seed':a.seed,'max_new_tokens':2048,'repetition_penalty':1.0,'temperature':0.8,'top_p':0.95,'top_k':0,
        'modes':a.modes,'world_size':world,'enable_thinking':False,'merged':False}
    a.output.mkdir(parents=True,exist_ok=True)
    if rank==0:
        rp=a.output/'inference_receipt.json'
        if rp.exists():assert json.loads(rp.read_text())==receipt,'Resume configuration changed'
        else:rp.write_text(json.dumps(receipt,indent=2))
    for mode in a.modes:
        path=a.output/f'{mode}.rank{rank}.jsonl'
        done={r['id'] for r in map(json.loads,path.read_text().splitlines())} if path.exists() else set()
        with path.open('a') as f,torch.inference_mode():
            for r in sorted(rows,key=lambda r:r['id'])[rank::world]:
                if r['id'] in done:continue
                def local(name):
                    assert name.startswith(a.media_from),name
                    q=a.media_root/name[len(a.media_from):];assert q.is_file(),str(q);return str(q)
                text=r['conversations'][0]['value']
                messages=[{'role':'system','content':[{'type':'text','text':r['system']}]},
                    {'role':'user','content':user_content(text,[local(n) for n in r.get('images',[])],[local(n) for n in r.get('videos',[])])}]
                inputs=p.apply_chat_template(copy.deepcopy(messages),tokenize=True,add_generation_prompt=True,
                    enable_thinking=False,return_dict=True,return_tensors='pt',
                    processor_kwargs={'fps':1.0,'do_sample_frames':True,'truncation':False}).to(device)
                inputs.pop('video_metadata',None)
                seed=int(hashlib.sha256(f'{a.seed}:{r["id"]}'.encode()).hexdigest()[:8],16)
                set_seed(seed)
                kw={'do_sample':mode=='sampling','max_new_tokens':2048,'repetition_penalty':1.0,
                    'eos_token_id':p.tokenizer.eos_token_id,'pad_token_id':p.tokenizer.pad_token_id,'use_cache':True}
                if mode=='sampling':kw.update(temperature=0.8,top_p=0.95,top_k=0)
                generated=model.generate(**inputs,**kw)
                ids=generated[0,inputs['input_ids'].shape[1]:]
                raw=p.decode(ids,skip_special_tokens=True).strip()
                result={'id':r['id'],'original_prompt':text.replace('<image>','').replace('<video>','').strip(),
                    'rewrite':raw,'checkpoint':'checkpoint-906','mode':mode,'seed':seed,
                    'generated_tokens':len(ids),'at_token_cap':len(ids)>=2048,
                    'last_token_id':int(ids[-1]) if len(ids) else None,
                    'rewrite_sha256':hashlib.sha256(raw.encode()).hexdigest()}
                f.write(json.dumps(result,ensure_ascii=False)+'\n');f.flush()
                print(mode,rank,r['id'],len(ids),flush=True)

if __name__=='__main__':main()
