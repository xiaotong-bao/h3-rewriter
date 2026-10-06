"""Warm every training source's requirement cache before GPU training."""
import argparse, concurrent.futures, json, pathlib, time, urllib.request
from run_config import PORT, JUDGE_REVISION
from reward_policy import validate_requirements

def precompute(inputs, port, workers, receipt):
    rows = [json.loads(line) for line in pathlib.Path(inputs).read_text().splitlines() if line.strip()]
    unique = {(r['original'], r['context']) for r in rows}
    def fetch(pair):
        source, context = pair
        request = urllib.request.Request(f'http://127.0.0.1:{port}/requirements',
            json.dumps({'original': source, 'context': context}).encode(), {'Content-Type': 'application/json'})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=1800) as response:
                    result = json.load(response)
                validate_requirements(source, result['requirements'])
                return len(result['requirements'])
            except Exception:
                if attempt == 2: raise
                time.sleep(5 * (attempt + 1))
    with urllib.request.urlopen(f'http://127.0.0.1:{port}/', timeout=5) as response:
        health = json.load(response)
    assert health.get('revision') == JUDGE_REVISION and health.get('runtime') == 'local_codex'
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        counts = []
        for n, count in enumerate(pool.map(fetch, sorted(unique)), 1):
            counts.append(count)
            print(f'REQUIREMENTS_READY {n}/{len(unique)}', flush=True)
    result = {'passed': True, 'judge_revision': JUDGE_REVISION, 'inputs': str(inputs),
              'unique_prompts': len(unique), 'requirements': sum(counts),
              'seconds': time.monotonic() - started, 'completed_at': time.time()}
    pathlib.Path(receipt).write_text(json.dumps(result, indent=2))
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=pathlib.Path, required=True)
    parser.add_argument('--port', type=int, default=PORT)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--receipt', type=pathlib.Path, required=True)
    args = parser.parse_args()
    assert 1 <= args.workers <= 16
    print(json.dumps(precompute(args.inputs, args.port, args.workers, args.receipt)), flush=True)
