"""Run the previously requested EP3 sampling benchmark after training releases GPUs."""
import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import time

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);a=p.parse_args()
    repo=pathlib.Path(__file__).resolve().parent.parent;root=a.root
    output=root/'ep3_sampling_101';output.mkdir(exist_ok=True)
    def state(phase,**extra):
        tmp=output/'status.json.tmp';tmp.write_text(json.dumps(dict(phase=phase,updated_at=time.time(),**extra),indent=2));tmp.replace(output/'status.json')
    job=root/'trl_sft_official_v2_20261006'
    state('waiting_for_training_completion')
    while True:
        status=json.loads((job/'pipeline_status.json').read_text())
        if status['phase']=='complete':
            raw=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
            if all(int(x.strip())<1024 for x in raw.splitlines()):break
            state('waiting_for_free_gpus')
        time.sleep(30)
    shutil.copy2(repo/'sft/regenerate_ep3_101.py',job/'regenerate_ep3_101.py')
    if not (output/'GENERATION_COMPLETE').exists():
        state('generating',checkpoint='checkpoint-906',mode='sampling',temperature=.8,top_p=.95,seed=20261006)
        cmd=['docker','run','--rm','--gpus','all','--network','none','--shm-size=16g',
            '-v',str(root)+':/work','-e','OMP_NUM_THREADS=1','h3-rewriter-sft-retrain:20261006',
            '/work/grpo_20261005/env/bin/python','-m','torch.distributed.run','--nproc_per_node=8','--master_port=29618',
            '/work/trl_sft_official_v2_20261006/regenerate_ep3_101.py',
            '--base','/work/models/Qwen3.5-9B','--adapter','/work/trl_sft_official_v2_20261006/run/checkpoint-906',
            '--inputs','/work/benchmark_jobs/arena_original_101_retention_v2_20261005/data/inputs.jsonl',
            '--system','/work/trl_sft_official_v2_20261006/system_prompt.txt',
            '--media-root','/work','--output','/work/ep3_sampling_101','--modes','sampling']
        with (output/'inference.log').open('a') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
        rows=[json.loads(s) for rank in range(8) for s in (output/f'sampling.rank{rank}.jsonl').read_text().splitlines()]
        assert len(rows)==len({r['id'] for r in rows})==101
        (output/'results.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in sorted(rows,key=lambda r:r['id'])))
        (output/'GENERATION_COMPLETE').write_text('101\n')
    rows=[json.loads(s) for s in (output/'results.jsonl').read_text().splitlines()]
    source=root/'benchmark_jobs/arena_original_101_retention_v2_20261005/data/inputs.jsonl'
    inputs={r['id']:r for r in map(json.loads,source.read_text().splitlines())};assert {r['id'] for r in rows}==set(inputs)
    manifest=[]
    for row in rows:
        src=inputs[row['id']];original=src['conversations'][0]['value'].replace('<image>','').strip();assert original==row['original_prompt']
        manifest.append(dict(id=row['id'],label='EP3_sampling',original=original,rewrite=row['rewrite'],
            task=src['task'],duration_s=src['duration_s'],aspect=src['aspect'],rewrite_sha256=hashlib.sha256(row['rewrite'].encode()).hexdigest()))
    (output/'manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in manifest))
    state('reviewing',model='gpt-6-astra',effort='low')
    with (output/'review.log').open('a') as log:
        subprocess.run([sys.executable,str(repo/'review/single_pass_compare.py'),'--manifest',str(output/'manifest.jsonl'),
            '--output',str(output),'--model','gpt-6-astra','--effort','low','--workers','8'],stdout=log,stderr=subprocess.STDOUT,check=True)
    summary=json.loads((output/'summary.json').read_text());assert not summary['failures'] and summary['models']['EP3_sampling']['valid']==101
    (output/'COMPLETE').write_text('101\n');state('complete',summary=summary['models']['EP3_sampling'])

if __name__=='__main__':main()
