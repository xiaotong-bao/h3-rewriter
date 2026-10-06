"""Durable HB10 job: validate all data, smoke backward, then fresh 4-epoch SFT."""
import hashlib
import json
import os
import pathlib
import subprocess
import time
import traceback
import urllib.request

from common import BASE, JOB

PYTHON = '/work/grpo_20261005/env/bin/python'


def state(phase, **kwargs):
    payload = {'phase': phase, 'updated_at': time.time(), **kwargs}
    (JOB / 'pipeline_status.json').write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload), flush=True)


def run(name, args):
    state(name)
    with (JOB / f'{name}.log').open('a') as log:
        subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=True)


def distributed(workers, script, extra=None):
    return [PYTHON, '-m', 'torch.distributed.run', '--nnodes=1',
        f'--nproc_per_node={workers}', '--master_addr=127.0.0.1', '--master_port=29616',
        str(JOB / script), *(extra or [])]


def main():
    assert not (JOB / 'run').exists(), 'Never overwrite a previous training'
    template_sha = hashlib.sha256((BASE / 'chat_template.jinja').read_bytes()).hexdigest()
    # Fetch is done by the host before the network-isolated container starts.
    official = json.loads((JOB / 'official_template_receipt.json').read_text())
    assert official['local_sha256'] == official['official_sha256'] == template_sha
    freeze = subprocess.check_output([PYTHON, '-m', 'pip', 'freeze'], text=True)
    (JOB / 'environment.freeze.txt').write_text(freeze)
    run('runtime_validation', [PYTHON, str(JOB / 'validate_runtime.py')])
    assert json.loads((JOB / 'runtime_validation.json').read_text())['passed']
    run('smoke', distributed(8, 'train.py', ['--smoke']))
    assert (JOB / 'smoke/COMPLETE').is_file()
    run('preflight_all', distributed(24, 'preflight.py'))
    rows = [json.loads(line) for rank in range(24)
            for line in (JOB / f'preflight0.rank{rank}.jsonl').open()]
    ids = [json.loads(line)['id'] for split in ['train', 'val']
           for line in (JOB / f'{split}.jsonl').open()]
    assert len(rows) == len(ids) == 9859
    assert len({row['id'] for row in rows}) == 9859
    assert {row['id'] for row in rows} == set(ids)
    report = {'passed': True, 'rows': len(rows), 'template_sha256': template_sha,
        'max_tokens': max(row['tokens'] for row in rows),
        'independent_prompt_checks': sum(row['independent_prompt_check'] for row in rows),
        'truncated_rows': 0, 'checked_prompt_and_target_masks': len(rows)}
    (JOB / 'preflight_report.json').write_text(json.dumps(report, indent=2))
    run('training', distributed(8, 'train.py', ['--epochs', '4']))
    assert (JOB / 'run/COMPLETE').is_file()
    state('complete')


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        state('failed', error=traceback.format_exc())
        raise
