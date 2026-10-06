"""Inventory or publish an explicit H3 data/SFT snapshot; never scan credentials."""
import argparse,concurrent.futures,hashlib,json,pathlib,time
import boto3
from boto3.s3.transfer import TransferConfig
parser=argparse.ArgumentParser()
parser.add_argument('--root',type=pathlib.Path,required=True)
parser.add_argument('--receipt',type=pathlib.Path,required=True)
parser.add_argument('--profile',default='r2w')
parser.add_argument('--endpoint',default='https://f25b0ac4c45a2442f62961145a64d158.r2.cloudflarestorage.com')
parser.add_argument('--upload',action='store_true',help='Explicitly upload the inventoried assets')
args=parser.parse_args();root=args.root
bucket='data-transfer-research';base='turboscale_migration_202603/xiaotong/h3_rewriter_sft_20261002/'
prefix=base+'trl_sft_official_v2_20261006/'
files=[]
def add(p,key):
 assert p.is_file(),str(p)
 files.append((p,key))
for folder in ('source','dataset','lf_dataset'):
 for p in sorted((root/folder).glob('*')):
  if p.is_file() and p.suffix in ('.jsonl','.json','.txt'):add(p,prefix+'data/'+folder+'/'+p.name)
sft=root/'trl_sft_official_v2_20261006'
for p in sorted(sft.glob('*')):
 if p.is_file() and (p.name in ('train.jsonl','val.jsonl','system_prompt.txt','data_report.json','pipeline_status.json','preflight_report.json','official_template_receipt.json','environment.freeze.txt','finalization_recovery.json') or p.name.startswith('epoch') and p.suffix in ('.json','.csv','.md')):add(p,prefix+'sft/'+p.name)
for step in (302,604,906,1208):
 folder=sft/f'run/checkpoint-{step}'
 required=['adapter_model.safetensors','adapter_config.json','optimizer.pt','scheduler.pt','trainer_state.json']+[f'rng_state_{i}.pth' for i in range(8)]
 assert all((folder/n).is_file() for n in required),'Incomplete checkpoint '+str(step)
 for p in sorted(folder.rglob('*')):
  if p.is_file():add(p,prefix+'sft/'+str(p.relative_to(sft)))
 for p in sorted((sft/f'epoch_benchmarks/step{step}').glob('*')):
  if p.is_file() and p.suffix in ('.json','.jsonl'):add(p,prefix+'sft/'+str(p.relative_to(sft)))
benchmark=root/'benchmark_jobs/arena_original_101_retention_v2_20261005/data/inputs.jsonl'
add(benchmark,prefix+str(benchmark.relative_to(root)))
refs=set()
for row in map(json.loads,benchmark.read_text().splitlines()):
 for name in row.get('images',[])+row.get('videos',[]):
  p=pathlib.Path(name.replace('/work/',str(root)+'/'));assert p.is_file(),str(p);refs.add(p)
for p in sorted(refs):add(p,prefix+str(p.relative_to(root)))
media=sorted(p for p in (root/'media').rglob('*') if p.is_file())
for p in media:add(p,base+'final/media/'+str(p.relative_to(root/'media')))
assert len({k for p,k in files})==len(files)
args.receipt.parent.mkdir(parents=True,exist_ok=True)
plan={'bucket':bucket,'prefix':prefix,'endpoint':args.endpoint,'files':len(files),'bytes':sum(p.stat().st_size for p,k in files),'media_files':len(media),'checkpoints':[302,604,906,1208],'source_rows':9899,'excluded_rows':40,'train_rows':9659,'val_rows':200,'uploaded':False}
args.receipt.write_text(json.dumps(plan,indent=2));print(json.dumps(plan),flush=True)
if not args.upload:raise SystemExit(0)
s3=boto3.Session(profile_name=args.profile).client('s3',endpoint_url=args.endpoint,region_name='auto')
remote={o['Key']:o['Size'] for page in s3.get_paginator('list_objects_v2').paginate(Bucket=bucket,Prefix=base+'final/media/') for o in page.get('Contents',[])}
def sha256(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def publish(pair):
 p,key=pair;size=p.stat().st_size;sha=sha256(p)
 # Legacy media inventory has no object SHA metadata: report its exact verification
 # level, rather than claiming a remote SHA check that was never performed.
 if '/final/media/' in key and remote.get(key)==size:
  return {'key':key,'bytes':size,'local_sha256':sha,'reused':True,'verification':'remote_key_and_size'}
 try:
  head=s3.head_object(Bucket=bucket,Key=key)
  reuse=head['ContentLength']==size and head.get('Metadata',{}).get('sha256')==sha
 except s3.exceptions.ClientError as e:
  if e.response['Error']['Code'] not in ('404','NoSuchKey','NotFound'):raise
  reuse=False
 if not reuse:s3.upload_file(str(p),bucket,key,ExtraArgs={'Metadata':{'sha256':sha}},Config=TransferConfig(max_concurrency=4))
 head=s3.head_object(Bucket=bucket,Key=key)
 assert head['ContentLength']==size and head['Metadata']['sha256']==sha
 return {'key':key,'bytes':size,'sha256':sha,'reused':reuse,'verification':'remote_size_and_sha256_metadata'}
receipts=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
 for i,r in enumerate(pool.map(publish,files),1):
  receipts.append(r)
  if i%100==0:print('VERIFIED',i,'/',len(files),flush=True)
plan.update(uploaded=True,ready=True,verified_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),artifacts=receipts)
args.receipt.write_text(json.dumps(plan,indent=2))
s3.put_object(Bucket=bucket,Key=prefix+'READY.json',Body=json.dumps(plan,indent=2).encode(),ContentType='application/json')
print('READY','s3://'+bucket+'/'+prefix,flush=True)
