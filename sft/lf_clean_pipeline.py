import json,subprocess,traceback,time
from pathlib import Path
root=Path('/work');job=root/'trl_sft_official_v2_20261006';py='/work/lf_env/bin/python'
def state(phase,**kw):
 data=dict(phase=phase,updated_at=time.time(),**kw);(job/'pipeline_status.json').write_text(json.dumps(data,indent=2));(root/'status.json').write_text(json.dumps(data,indent=2));print(phase,flush=True)
def run(phase,cmd):
 state(phase)
 with (root/'logs'/(phase+'.log')).open('a') as log:subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
def ddp(script,config):return [py,'-m','torch.distributed.run','--nnodes=1','--nproc_per_node=8','--master_addr=127.0.0.1','--master_port=29624',script,config]
try:
 assert not (root/'runs/content_clean_9bv2').exists()
 if not (root/'logs/lf_preflight_report.json').exists():run('preflight',[py,'/work/derive_validated_cache.py'])
 report=json.loads((root/'logs/lf_preflight_report.json').read_text());assert report['train']['rows']==5682 and report['val']['rows']==200
 assert all(x['target_mismatch_count']==0 for x in report.values())
 run('smoke',ddp('/work/LLaMA-Factory/src/llamafactory/launcher.py','/work/lf_configs/smoke.yaml'))
 assert json.loads((root/'lf_configs/sft.yaml').read_text())['num_train_epochs']==6
 run('training',ddp('/work/train_with_epoch_review.py','/work/lf_configs/sft.yaml'))
 assert json.loads((job/'progress.json').read_text())['epoch']>=6
 (root/'COMPLETE').write_text('complete\n');state('complete')
except BaseException:state('failed',error=traceback.format_exc());raise
