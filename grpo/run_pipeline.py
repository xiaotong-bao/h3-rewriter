import json,os,pathlib,subprocess,sys,time,urllib.request
ROOT=pathlib.Path('/work/grpo_ep3_luna_v2_20261006')
def state(stage,**kw):
 value={'stage':stage,'time':time.time(),**kw}
 (ROOT/'pipeline_status.json').write_text(json.dumps(value,indent=2));print(json.dumps(value),flush=True)
authorization=json.loads((ROOT/'direct_grpo_authorization.json').read_text())
assert authorization['approved']
port=int(os.environ.get('H3_LUNA_PORT','8792'))
with urllib.request.urlopen(f'http://127.0.0.1:{port}/',timeout=5) as response:judge_health=json.load(response)
assert judge_health.get('ready') and judge_health.get('revision')=='ep3-v2-luna-severity-v3' and judge_health.get('model')=='gpt-6-luna','Local Luna gateway is not ready'
while not (ROOT/'MERGE_READY').is_file():time.sleep(10)
(ROOT/'environment.freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
for stage,flags in [('smoke',['--smoke']),('pilot',['--steps','64'])]:
 if (ROOT/stage/'COMPLETE').exists():continue
 checkpoints=sorted((ROOT/stage).glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]))
 if checkpoints:flags+=['--resume',str(checkpoints[-1])]
 state(stage,state='running')
 cmd=[sys.executable,'-m','torch.distributed.run','--nproc_per_node=8','--master_addr=127.0.0.1','--master_port=29708',str(ROOT/'train_grpo.py'),*flags]
 with (ROOT/(stage+'.log')).open('a') as log:result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
 if result.returncode:
  state(stage,state='failed',exit_code=result.returncode);raise SystemExit(result.returncode)
 assert (ROOT/stage/'COMPLETE').is_file()
state('evaluation_101',state='running')
cmd=[sys.executable,'-m','torch.distributed.run','--nproc_per_node=8','--master_addr=127.0.0.1','--master_port=29708',str(ROOT/'evaluate.py')]
with (ROOT/'evaluation_101.log').open('a') as log:result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
state('evaluation_101',state='complete' if result.returncode==0 else 'failed',exit_code=result.returncode)
raise SystemExit(result.returncode)
