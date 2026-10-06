"""Recover final export when the duplicate final benchmark callback failed.

Does not perform optimizer steps. Validates the completed checkpoint and 101
epoch-4 generations before copying the exact adapter and processor artifacts.
"""
import hashlib
import json
import shutil
import time
from transformers import AutoConfig, GenerationConfig
from common import BASE, JOB, processor

run = JOB / 'run'
checkpoint = run / 'checkpoint-1208'
state = json.loads((checkpoint / 'trainer_state.json').read_text())
assert state['global_step'] == state['max_steps'] == 1208
assert abs(state['epoch'] - 4) < 1e-6
benchmark = JOB / 'epoch_benchmarks/step1208'
assert (benchmark / 'COMPLETE').is_file()
rows = [json.loads(s) for s in (benchmark / 'results.jsonl').read_text().splitlines()]
assert len(rows) == len({r['id'] for r in rows}) == 101
assert all(r['step'] == 1208 and r['epoch'] == 4 for r in rows)
for e in (2,3):
    destination = JOB / 'epoch_benchmarks' / f'step{e*302}'
    for ext in ('json','csv','md'):
        source = JOB / f'epoch{e}_text_audit_101.{ext}'
        target = destination / source.name
        shutil.copy2(source,target)
        assert source.read_bytes() == target.read_bytes()
    report = json.loads((JOB / f'epoch{e}_text_audit_101.json').read_text())
    summary = json.loads((destination / 'summary.json').read_text())
    summary.update(text_audit_status='complete_case_by_case_review',
                   text_audit_counts=report['metadata']['counts'],
                   format_structure_passed=report['metadata']['format_structure_passed'])
    (destination / 'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
final = run / 'final_adapter'
assert not final.exists(), 'Never overwrite a final adapter'
final.mkdir()
for path in checkpoint.iterdir():
    if path.name in ('adapter_model.safetensors','adapter_config.json','README.md'):
        shutil.copy2(path,final/path.name)
original = checkpoint / 'adapter_model.safetensors'
copied = final / original.name
digest = hashlib.sha256(original.read_bytes()).hexdigest()
assert hashlib.sha256(copied.read_bytes()).hexdigest() == digest
p = processor()
p.save_pretrained(final)
config = GenerationConfig.from_model_config(AutoConfig.from_pretrained(BASE,local_files_only=True))
config.eos_token_id = p.tokenizer.eos_token_id
config.pad_token_id = p.tokenizer.pad_token_id
config.save_pretrained(final)
receipt = {'phase':'complete','updated_at':time.time(),
    'recovered_from':'checkpoint-1208','optimizer_steps':1208,'epoch':4,
    'epoch4_benchmark_rows':101,'adapter_sha256':digest,
    'recovery_reason':'Duplicate final on_save attempted to overwrite already completed benchmark.',
    'additional_optimizer_steps':0}
(JOB/'finalization_recovery.json').write_text(json.dumps(receipt,indent=2))
(run/'COMPLETE').write_text('complete\n')
(JOB/'pipeline_status.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
