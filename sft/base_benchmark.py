"""Queue the untouched base benchmark after training releases GPU memory."""
import json
import time
from types import SimpleNamespace

import torch
from transformers import Qwen3_5ForConditionalGeneration

from common import BASE, JOB, processor
from epoch_benchmark import EpochBenchmark

status = JOB / 'base_benchmark_status.json'
status.write_text(json.dumps({'phase': 'waiting_for_training', 'model': str(BASE)}))
while not (JOB / 'run/COMPLETE').exists():
    time.sleep(30)
time.sleep(30)
torch.set_num_threads(1)
p = processor()
status.write_text(json.dumps({'phase': 'loading_base', 'model': str(BASE)}))
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    BASE, dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda:0')
EpochBenchmark(p).on_save(None, SimpleNamespace(epoch=0, global_step=0), None, model=model)
path = JOB / 'epoch_benchmarks/step0/results.jsonl'
rows = [json.loads(line) for line in path.open()]
for row in rows:
    row['checkpoint'] = 'original_Qwen3.5-9B_no_adapter'
path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
status.write_text(json.dumps({'phase': 'complete', 'rows': len(rows), 'model': str(BASE),
    'text_audit_status': 'pending_case_by_case_review'}))
