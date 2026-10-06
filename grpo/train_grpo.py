"""TRL GRPO pilot: EP3 initialization, frozen Codex GPT-6 Luna judge."""
import argparse,concurrent.futures,json,math,os,pathlib,time,urllib.request
import torch
from datasets import Dataset
from peft import LoraConfig,get_peft_model
from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration,TrainerCallback
from trl import GRPOConfig,GRPOTrainer

ROOT=pathlib.Path('/work/grpo_ep3_luna_v2_20261006')
def clean_nulls(value):
    # Arrow unifies image/text block structs and introduces null keys. Qwen's
    # template checks key presence, so null image keys must be removed.
    if isinstance(value,dict):return {k:clean_nulls(v) for k,v in value.items() if v is not None}
    if isinstance(value,list):return [clean_nulls(v) for v in value]
    return value
def clean_batch(batch):return clean_nulls(batch)
def call(payload):
    with (ROOT/f'candidate_requests.rank{os.environ.get("RANK","0")}.jsonl').open('a') as f:
        f.write(json.dumps(payload,ensure_ascii=False)+'\n')
    port=int(os.environ.get("H3_LUNA_PORT","8792"))
    req=urllib.request.Request(f'http://127.0.0.1:{port}/score',json.dumps(payload).encode(),{'Content-Type':'application/json'})
    errors=[]
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req,timeout=1800) as response:
                result=json.load(response)
            assert isinstance(result.get('reward'),(int,float)) and math.isfinite(result['reward'])
            return result
        except Exception as error:
            detail=error.read().decode(errors='replace') if isinstance(error,urllib.error.HTTPError) else repr(error)
            errors.append(detail)
            with (ROOT/f'judge_failures.rank{os.environ.get("RANK","0")}.jsonl').open('a') as log:
                log.write(json.dumps({'time':time.time(),'attempt':attempt+1,'error':detail,'payload':payload},ensure_ascii=False)+'\n')
            if attempt<2:time.sleep(5*(attempt+1))
    # This is an operational penalty, never a fabricated semantic verdict.
    return {'reward':-1.,'retention':None,'items':[], 'judge_failed':True,
            'failure_penalty':-1.,'attempts':3,'errors':errors}
def completion_text(value):
    if isinstance(value,str):return value
    content=value[-1]['content']
    if isinstance(content,str):return content
    return ''.join(x.get('text','') for x in content)
def retention_reward(completions,original,context,task,trainer_state,**kwargs):
    rewards=[]
    texts=[completion_text(value) for value in completions]
    payloads=[{'original':source,'context':ctx,'rewrite':text} for text,source,ctx in zip(texts,original,context,strict=True)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        verdicts=list(pool.map(call,payloads))
    for text,source,ctx,t,result in zip(texts,original,context,task,verdicts,strict=True):
        sections=['integrated_multimodal_description','overall_soundscape','non_diegetic_music']
        assert t in ('i2va','t2va')
        import re
        from format_rules import check_format
        duration=float(re.search(r'duration:\s*([0-9.]+)s',ctx)[1])
        checks=check_format(text,t,duration)
        valid=all(checks.values())
        reward=result['reward'] if valid else result['reward']-.3
        rewards.append(reward)
        with (ROOT/f'rollouts.rank{os.environ.get("RANK","0")}.jsonl').open('a') as f:
            f.write(json.dumps({'step':trainer_state.global_step,'original':source,'rewrite':text,
                'reward':reward,'format_valid':valid,'format_checks':checks,'verdict':result},ensure_ascii=False)+'\n')
    return rewards

class Status(TrainerCallback):
    def on_log(self,args,state,control,logs=None,**kwargs):
        if state.is_world_process_zero:
            (pathlib.Path(args.output_dir)/'status.json').write_text(json.dumps({'step':state.global_step,'max_steps':state.max_steps,'logs':logs},indent=2))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');parser.add_argument('--steps',type=int,default=64);parser.add_argument('--resume',type=str)
    a=parser.parse_args();torch.manual_seed(42)
    authorization=json.loads((ROOT/'direct_grpo_authorization.json').read_text())
    assert authorization['approved'] and authorization['judge_revision']=='ep3-v2-luna-severity-v3'
    assert (ROOT/'MERGE_READY').is_file()
    rows=[json.loads(x) for x in (ROOT/'pilot_inputs.jsonl').read_text().splitlines()]
    if a.smoke:
        rows=[next(r for r in rows if r['task']=='t2va'),next(r for r in rows if r['task']=='i2va')]
    dataset=Dataset.from_list(rows).with_transform(clean_batch)
    processor=AutoProcessor.from_pretrained(ROOT/'sft_init')
    processor.image_processor.size={'shortest_edge':3136,'longest_edge':200704}
    # Verify true image-token expansion rather than silently truncating inputs.
    for row in dataset:
        encoded=processor.apply_chat_template(row['prompt'],tokenize=True,add_generation_prompt=True,
            enable_thinking=False,return_dict=True)
        ids=encoded['input_ids']
        if ids and isinstance(ids[0],list):ids=ids[0]
        assert len(ids)<=6144,f'Prompt too long: {row["id"]}'
    model=Qwen3_5ForConditionalGeneration.from_pretrained(ROOT/'sft_init',dtype=torch.bfloat16,attn_implementation='sdpa')
    targets=r'.*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)'
    config=LoraConfig(r=32,lora_alpha=64,lora_dropout=0.,bias='none',task_type='CAUSAL_LM',target_modules=targets)
    assert processor.tokenizer.eos_token_id==248046
    model.generation_config.eos_token_id=processor.tokenizer.eos_token_id
    model.generation_config.pad_token_id=processor.tokenizer.pad_token_id
    model=get_peft_model(model,config);model.enable_input_require_grads()
    assert all('language_model' in n for n,p in model.named_parameters() if p.requires_grad)
    model.print_trainable_parameters()
    output=ROOT/('smoke' if a.smoke else 'pilot')
    if a.resume:
        checkpoint=pathlib.Path(a.resume)
        assert checkpoint.parent==output and (checkpoint/'trainer_state.json').is_file(), 'Invalid resume checkpoint'
    else:assert not output.exists(),f'Output already exists: {output}'
    args=GRPOConfig(output_dir=str(output),learning_rate=5e-6,max_steps=2 if a.smoke else a.steps,
        per_device_train_batch_size=1,gradient_accumulation_steps=2,num_generations=8,
        max_completion_length=2048,beta=.02,temperature=.8,top_p=.95,loss_type='dapo',
        scale_rewards='batch',mask_truncated_completions=True,bf16=True,
        gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
        ddp_find_unused_parameters=False,remove_unused_columns=False,
        logging_steps=1,save_steps=16,save_total_limit=None,report_to='none',
        warmup_steps=2,seed=42,data_seed=42,chat_template_kwargs={'enable_thinking':False},
        generation_kwargs={'use_cache':True,'eos_token_id':processor.tokenizer.eos_token_id,'pad_token_id':processor.tokenizer.pad_token_id},log_completions=False,log_multimodal=False)
    trainer=GRPOTrainer(model=model,args=args,processing_class=processor,
        train_dataset=dataset,reward_funcs=retention_reward,callbacks=[Status()])
    trainer.train(resume_from_checkpoint=a.resume);trainer.save_model(str(output/'final_adapter'))
    if trainer.is_world_process_zero():
        processor.save_pretrained(output/'final_adapter')
        (output/'COMPLETE').write_text(str(trainer.state.global_step)+'\n')
        print('GRPO_COMPLETE',trainer.state.global_step,flush=True)
if __name__=='__main__':main()
