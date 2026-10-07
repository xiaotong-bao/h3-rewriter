"""Read-only R2 restoration for a new SFT run. Never restores old adapters."""
import argparse
import concurrent.futures
import hashlib
import json
import pathlib
import tarfile
import time

import boto3
from boto3.s3.transfer import TransferConfig

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);a=p.parse_args()
    root=a.root;root.mkdir(parents=True,exist_ok=True)
    def state(phase,**extra):
        payload=dict(phase=phase,updated_at=time.time(),**extra)
        (root/'restore_status.json').write_text(json.dumps(payload,indent=2));print(json.dumps(payload),flush=True)
    c=boto3.Session(profile_name='r2w').client('s3',endpoint_url='https://f25b0ac4c45a2442f62961145a64d158.r2.cloudflarestorage.com',region_name='auto')
    bucket='data-transfer-research';base='turboscale_migration_202603/xiaotong/h3_rewriter_sft_20261002/'
    backups='turboscale_migration_202603/xiaotong/h3_rewriter_sft_retention_v2_20261006/assets/'
    ready=json.loads(c.get_object(Bucket=bucket,Key=backups+'READY.json')['Body'].read())
    (root/'source_asset_READY.json').write_text(json.dumps(ready,indent=2))
    transfer=TransferConfig(max_concurrency=8)
    def fetch(key,path,size=None,digest=None):
        path.parent.mkdir(parents=True,exist_ok=True)
        if not path.is_file() or size is not None and path.stat().st_size!=size:
            partial=path.with_suffix(path.suffix+'.partial')
            c.download_file(bucket,key,str(partial),Config=transfer);partial.replace(path)
        if size is not None:assert path.stat().st_size==size,str(path)
        if digest:assert sha(path)==digest,'SHA mismatch '+str(path)
    state('restore_datasets')
    for name,count in [('train.jsonl',9659),('val.jsonl',200)]:
        path=root/'lf_dataset'/name;fetch(base+'training_bundle/lf_dataset/'+name,path)
        rows=[json.loads(s) for s in path.read_text().splitlines()];assert len(rows)==count
    for folder,name in [('dataset','split_report.json'),('dataset','system_prompt.txt'),('dataset','val.jsonl'),('dataset','train_partial.jsonl')]:
        fetch(base+'training_bundle/'+folder+'/'+name,root/folder/name)
    media={}
    for split in ('train','val'):
        for row in map(json.loads,(root/'lf_dataset'/f'{split}.jsonl').read_text().splitlines()):
            for original in row.get('images',[])+row.get('videos',[]):
                assert original.startswith('/work/media/'),original
                relative=original[len('/work/media/'):]
                path=pathlib.PurePosixPath(relative);assert not path.is_absolute() and '..' not in path.parts
                media[base+'final/media/'+relative]=root/'media'/relative
    state('inventory_media',unique_media=len(media))
    sizes={o['Key']:o['Size'] for page in c.get_paginator('list_objects_v2').paginate(Bucket=bucket,Prefix=base+'final/media/') for o in page.get('Contents',[]) if o['Key'] in media}
    assert set(sizes)==set(media),'Missing remote training media'
    media_bytes=sum(sizes.values())
    state('restore_media',unique_media=len(media),bytes=media_bytes)
    def restore(pair):
        key,path=pair;fetch(key,path,sizes[key]);return key
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
        for i,_ in enumerate(ex.map(restore,media.items()),1):
            if i%100==0:state('restore_media',done=i,total=len(media),bytes=media_bytes)
    for name in ('base','env'):
        asset=next(x for x in ready['packages'] if x['name']==name)
        archive=root/'archives'/f'{name}.tar'
        state('restore_'+name,bytes=asset['bytes'])
        fetch(asset['key'],archive,asset['bytes'],asset['sha256'])
        # Inspect first, then extract into an isolated subfolder. Absolute or
        # escaping archive paths are rejected by the data filter.
        target=root/'restored'/name;marker=target/'EXTRACTED.json'
        if not marker.exists():
            target.mkdir(parents=True,exist_ok=True)
            with tarfile.open(archive) as t:
                names=[m.name for m in t.getmembers()]
                (root/f'{name}_archive_members.json').write_text(json.dumps(names,indent=2))
                t.extractall(target,filter='data')
            marker.write_text(json.dumps(asset,indent=2))
    state('assets_restored',unique_media=len(media),media_bytes=media_bytes,
        note='Training has not started; template/env verification and smoke/preflight still required.')

if __name__=='__main__':main()
