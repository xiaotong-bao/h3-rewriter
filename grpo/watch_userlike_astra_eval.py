"""Queue frozen 101-case Astra evaluations for every 32-step checkpoint."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import threading
import concurrent.futures

ROOT = Path('/data/xiaotong/h3_rewriter_sft_20261002')
JOB = ROOT / os.environ.get('H3_GRPO_JOB', 'grpo_9bv2_userlike_1k_20261007')
INTERVAL = int(os.environ.get('H3_EVAL_INTERVAL', '32'))
OUT = JOB / f'astra_eval_every{INTERVAL}'
REPO = Path('/home/xiaotong/h3-rewriter')


def manifest_row(row, sources):
    source = sources[row['id']]
    original = source['conversations'][0]['value'].replace('<image>', '').strip()
    assert original == row['original_prompt']
    system = source['system']
    return dict(id=row['id'], label=row['label'], original=original,
        rewrite=row['rewrite'], task=re.search(r'Requested task:\s*(\w+)', system)[1],
        duration_s=float(re.search(r'duration:\s*([0-9.]+)s', system)[1]),
        aspect=re.search(r'aspect ratio:\s*([^;]+)', system)[1],
        rewrite_sha256=hashlib.sha256(row['rewrite'].encode()).hexdigest())


def stream_grade(output, stop):
    sys.path.insert(0, str(REPO / 'review'))
    import single_pass_compare as judge
    sources = {r['id']: r for r in map(json.loads, (ROOT / 'benchmark_jobs/arena_original_101_retention_v2_20261005/data/inputs.jsonl').read_text().splitlines())}
    seen, futures, valid, failures = set(), {}, 0, []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        while True:
            for path in (output / 'generation').glob('rank*.jsonl'):
                for line in path.read_text().splitlines():
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if raw['id'] in seen:
                        continue
                    row = manifest_row(raw, sources)
                    seen.add(row['id'])
                    futures[pool.submit(judge.grade_retry, row, output, 'gpt-6-astra', 'low', judge.REVISION)] = row['id']
            for future in list(futures):
                if not future.done():
                    continue
                sample = futures.pop(future)
                try:
                    future.result()
                    valid += 1
                except Exception as error:
                    failures.append(dict(id=sample, error=repr(error)))
            write_json(output / 'stream_status.json', dict(generated=len(seen), graded=valid, in_flight=len(futures), failures=failures, expected=101))
            if stop.is_set() and not futures:
                return
            stop.wait(5) if not stop.is_set() else time.sleep(1)


def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def evaluate(step, checkpoint):
    output = OUT / f'step-{step:04d}'
    output.mkdir(exist_ok=True)
    container = (f'xiaotong-9bv2-full-astra-step{step}-20261008' if INTERVAL == 100 else
                 f'xiaotong-9bv2-astra-eval-step{step}-20261007')
    generated = output / 'generation'
    if not (generated / 'COMPLETE').exists():
        inspect = subprocess.run(['docker', 'inspect', '--format', '{{.State.Status}}', container], capture_output=True, text=True)
        if inspect.returncode == 0 and inspect.stdout.strip() != 'running':
            subprocess.run(['docker', 'rm', container], check=True)
            inspect = None
        if inspect is None or inspect.returncode != 0:
            command = ['docker', 'run', '-d', '--name', container, '--gpus', 'all',
                '--network', 'none', '--shm-size=8g', '-v', f'{ROOT}:/work',
                '-v', f'{REPO}:/code:ro', '-e', 'HF_HUB_OFFLINE=1', '-e', 'OMP_NUM_THREADS=1',
                '-e', 'PYTHONPATH=/work/trl_sft_official_v2_20261006/vendor:/work/grpo_20261005/env/lib/python3.11/site-packages',
                'h3-rewriter-grpo:20261005', '/work/grpo_20261005/env/bin/python', '-u',
                '-m', 'torch.distributed.run', '--nproc_per_node=8', '--master_port=29852',
                '/code/grpo/ablate_9bv2_inputs.py', '--output', str(Path('/work') / generated.relative_to(ROOT)),
                '--checkpoint', str(Path('/work') / checkpoint.relative_to(ROOT)),
                '--templates', 'HF', '--merges', 'unmerged', '--autocast',
                '--label-suffix', f'GRPO_step{step}']
            subprocess.run(command, check=True)
        write_json(OUT / 'status.json', dict(stage='generating', step=step, output=str(output), container=container))
        stop = threading.Event()
        grader = threading.Thread(target=stream_grade, args=(output, stop), daemon=True)
        grader.start()
        with (output / 'generation.log').open('a') as log:
            waited = subprocess.run(['docker', 'wait', container], stdout=subprocess.PIPE, text=True)
            subprocess.run(['docker', 'logs', container], stdout=log, stderr=subprocess.STDOUT)
        stop.set()
        grader.join()
        assert waited.returncode == 0 and waited.stdout.strip() == '0', f'Generation failed: {container}'
        assert (generated / 'COMPLETE').exists()
    sources = {r['id']: r for r in map(json.loads, (ROOT / 'benchmark_jobs/arena_original_101_retention_v2_20261005/data/inputs.jsonl').read_text().splitlines())}
    rows = [json.loads(s) for s in (generated / 'results.jsonl').read_text().splitlines()]
    assert len(rows) == len({r['id'] for r in rows}) == 101
    manifest = [manifest_row(row, sources) for row in rows]
    (output / 'manifest.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in manifest))
    write_json(OUT / 'status.json', dict(stage='astra_grading', step=step, output=str(output)))
    with (output / 'astra.log').open('a') as log:
        subprocess.run([sys.executable, str(REPO / 'review/single_pass_compare.py'),
            '--manifest', str(output / 'manifest.jsonl'), '--output', str(output),
            '--model', 'gpt-6-astra', '--effort', 'low', '--workers', '8'],
            stdout=log, stderr=subprocess.STDOUT, check=True)
    summary = json.loads((output / 'summary.json').read_text())
    assert not summary['failures'] and all(s['valid'] == s['expected'] == 101 for s in summary['models'].values())
    (output / 'COMPLETE').write_text('101\n')
    rebuild_report()


def rebuild_report():
    results = []
    baseline = Path('/home/xiaotong/9bv2_grpo_amp_ablation_20261007/summary.json')
    if INTERVAL == 32 and baseline.exists():
        summary = json.loads(baseline.read_text())
        assert not summary['failures'] and all(s['valid'] == s['expected'] == 101 for s in summary['models'].values())
        results.append(dict(step=0, **summary))
    for path in sorted(OUT.glob('step-*')):
        if (path / 'COMPLETE').exists():
            results.append(dict(step=int(path.name.split('-')[1]), **json.loads((path / 'summary.json').read_text())))
    write_json(OUT / 'results.json', results)
    table = ['| Step | Valid | Content | Format | Severe | General | None |', '|---:|---:|---:|---:|---:|---:|---:|']
    for row in results:
        for s in row['models'].values():
            table.append(f"| {row['step']} | {s['valid']}/101 | {s['content_score']:.2f} | {s['format_score']:.2f} | {s['severe']} | {s['general']} | {s['none']} |")
    (OUT / 'COMPARISON.md').write_text('\n'.join(table) + '\n')


def main():
    OUT.mkdir(exist_ok=True)
    lock = (OUT / 'watcher.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (OUT / 'watcher.pid').write_text(str(os.getpid()) + '\n')
    while True:
        pending = []
        for checkpoint in (JOB / 'pilot').glob('checkpoint-*'):
            step = int(checkpoint.name.split('-')[1])
            marker = checkpoint / 'trainer_state.json'
            if step % INTERVAL == 0 and marker.exists() and time.time() - marker.stat().st_mtime > 60:
                pending.append((step, checkpoint))
        if (JOB / 'pilot/COMPLETE').exists():
            pending.append((int((JOB / 'pilot/COMPLETE').read_text()), JOB / 'pilot/final_adapter'))
        for step, checkpoint in sorted(pending):
            if (OUT / f'step-{step:04d}/COMPLETE').exists():
                continue
            try:
                evaluate(step, checkpoint)
            except Exception as error:
                print('EVAL_RETRY', step, repr(error), flush=True)
                write_json(OUT / 'status.json', dict(stage='retry_pending', step=step, error=repr(error)))
                time.sleep(60)
                break
        else:
            write_json(OUT / 'status.json', dict(stage='waiting_for_checkpoint', completed_steps=[int(p.parent.name.split('-')[1]) for p in sorted(OUT.glob('step-*/COMPLETE'))]))
            if (JOB / 'pilot/COMPLETE').exists():
                (OUT / 'COMPLETE').write_text('all scheduled evaluations complete\n')
                return
        time.sleep(15)


if __name__ == '__main__':
    main()
