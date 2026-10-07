"""Review each completed 101-case epoch benchmark using the frozen Astra rubric."""
import argparse
import fcntl
import hashlib
import json
import pathlib
import subprocess
import sys
import time

import single_pass_compare as judge

def read(path):
    return json.loads(path.read_text()) if path.exists() else None

def write(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2));temp.replace(path)

def build_manifest(benchmark,source,output,epoch):
    inputs={r['id']:r for r in map(json.loads,source.read_text().splitlines())}
    results=[json.loads(s) for s in (benchmark/'results.jsonl').read_text().splitlines()]
    assert len(inputs)==len(results)==len({r['id'] for r in results})==101
    assert {r['id'] for r in results}==set(inputs),'Epoch benchmark ID mismatch'
    rows=[]
    for row in sorted(results,key=lambda r:r['id']):
        src=inputs[row['id']]
        original=src['conversations'][0]['value'].replace('<image>','').replace('<video>','').strip()
        assert row['original_prompt']==original,'Original prompt mismatch'
        assert abs(row['epoch']-epoch)<1e-6
        rows.append(dict(id=row['id'],label=f'EP{epoch}_greedy',original=original,rewrite=row['rewrite'],
            task=src['task'],duration_s=src['duration_s'],aspect=src['aspect'],
            rewrite_sha256=hashlib.sha256(row['rewrite'].encode()).hexdigest(),
            checkpoint=row['checkpoint'],epoch=epoch,step=row['step'],
            at_token_cap=row['at_token_cap']))
    contents=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)
    manifest=output/'manifest.jsonl'
    if manifest.exists():assert manifest.read_text()==contents,'Immutable epoch manifest changed'
    else:manifest.write_text(contents)
    return manifest

def refresh_table(root,epochs):
    states={};table=['# 各 epoch 的标准 101 评审','',
        f'评审：gpt-6-astra / low；固定规则 `{judge.REVISION}`。所有分数为同一批 101 原文上的自动评审分。','',
        '| 轮次 | 状态 | 内容分 | 格式分 | 严重 case | 一般 case | 无问题 case |',
        '|---|---|---:|---:|---:|---:|---:|']
    reference=read(root/'reference_summary.json')
    if reference:
        assert reference['revision']==judge.REVISION
        for label in ('fal','9bv2'):
            m=reference['models'][label]
            assert m['valid']==m['expected']==101
            table.append(f'| {label}（固定参考） | 101/101 | {m["content_score"]:.2f} | {m["format_score"]:.2f} | {m["severe"]} | {m["general"]} | {m["none"]} |')
    for epoch in range(1,epochs+1):
        folder=root/f'ep{epoch}';s=read(folder/'summary.json')
        if (folder/'COMPLETE').exists():
            m=s['models'][f'EP{epoch}_greedy'];states[f'ep{epoch}']=dict(status='complete',**m)
            table.append(f'| EP{epoch} | 101/101 | {m["content_score"]:.2f} | {m["format_score"]:.2f} | {m["severe"]} | {m["general"]} | {m["none"]} |')
        else:
            state=read(folder/'state.json') or {'status':'waiting_for_epoch_outputs'}
            states[f'ep{epoch}']=state;table.append(f'| EP{epoch} | {state["status"]} | — | — | — | — | — |')
    (root/'EPOCH_COMPARISON.md').write_text('\n'.join(table)+'\n')
    write(root/'status.json',dict(updated_at=time.time(),model='gpt-6-astra',effort='low',revision=judge.REVISION,epochs=states))

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=pathlib.Path,required=True)
    p.add_argument('--epochs',type=int,default=5);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--poll-seconds',type=int,default=30);p.add_argument('--once',action='store_true');a=p.parse_args()
    root=a.run_root/'epoch_reviews';root.mkdir(exist_ok=True)
    lock=(root/'watcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    job=a.run_root/'trl_sft_official_v2_20261006'
    source=a.run_root/'benchmark_jobs/arena_original_101_retention_v2_20261005/data/inputs.jsonl'
    write(root/'rubric_receipt.json',dict(revision=judge.REVISION,model='gpt-6-astra',effort='low',
        prompt_sha256=hashlib.sha256(judge.RULES.encode()).hexdigest(),
        schema_sha256=hashlib.sha256(json.dumps(judge.SCHEMA,sort_keys=True).encode()).hexdigest(),
        benchmark_inputs_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    while True:
        refresh_table(root,a.epochs)
        for benchmark in sorted((job/'epoch_benchmarks').glob('step*'),key=lambda p:int(p.name[4:])):
            if not (benchmark/'COMPLETE').exists():continue
            result=json.loads((benchmark/'results.jsonl').read_text().splitlines()[0]);epoch=round(result['epoch'])
            assert 1<=epoch<=a.epochs
            output=root/f'ep{epoch}';output.mkdir(exist_ok=True)
            if (output/'COMPLETE').exists():continue
            previous=read(output/'state.json') or {};attempts=previous.get('attempts',0)
            if attempts>=3 or time.time()-previous.get('failed_at',0)<300:continue
            try:
                manifest=build_manifest(benchmark,source,output,epoch)
                write(output/'state.json',dict(status='reviewing',attempts=attempts+1,started_at=time.time(),checkpoint=result['checkpoint']))
                refresh_table(root,a.epochs)
                print('REVIEW_START',epoch,flush=True)
                command=[sys.executable,str(pathlib.Path(judge.__file__)),
                    '--manifest',str(manifest),'--output',str(output),'--model','gpt-6-astra','--effort','low',
                    '--revision',judge.REVISION,'--workers',str(a.workers)]
                with (output/'review.log').open('a') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
                summary=read(output/'summary.json');group=summary['models'][f'EP{epoch}_greedy']
                assert not summary['failures'] and group['valid']==group['expected']==101
                (output/'COMPLETE').write_text('101\n')
                write(output/'state.json',dict(status='complete',attempts=attempts+1,completed_at=time.time(),checkpoint=result['checkpoint']))
                print('REVIEW_COMPLETE',epoch,json.dumps(group),flush=True)
            except Exception as e:
                write(output/'state.json',dict(status='failed',attempts=attempts+1,failed_at=time.time(),error=str(e)))
                print('REVIEW_FAILED',epoch,str(e),flush=True)
            refresh_table(root,a.epochs)
        if a.once or all((root/f'ep{i}/COMPLETE').exists() for i in range(1,a.epochs+1)):break
        time.sleep(a.poll_seconds)

if __name__=='__main__':main()
