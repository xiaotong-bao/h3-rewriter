"""Select an unchanged subset of the fully checked native LLaMA-Factory cache."""
import json,hashlib
from pathlib import Path
from datasets import load_from_disk,DatasetDict
from llamafactory.hparams import get_train_args
from llamafactory.model import load_tokenizer
root=Path('/work');parent=Path('/parent_dataset');proof=Path('/parent_checks')
visual=json.loads((proof/'lf_visual_audit_report.json').read_text());gate=json.loads((proof/'lf_preflight_report.json').read_text())
assert visual['total_rows']==19718 and visual['visual_rows_checked']==visual['visual_rows']==16720 and not visual['errors']
assert gate['train']['rows']==19318 and gate['val']['rows']==400 and all(x['target_mismatch_count']==0 for x in gate.values())
cache=load_from_disk(str(parent/'tokenized_v2'));cfg=json.loads((root/'lf_configs/sft.yaml').read_text());ma,da,ta,fa,ga=get_train_args(dict(cfg,use_cpu=True));tokenizer=load_tokenizer(ma)['tokenizer'];report={};derived={}
for key,ds in cache.items():
 split='train' if key=='train' else 'val';expected=5682 if split=='train' else 200
 old=list(map(json.loads,(parent/(split+'.jsonl')).read_text().splitlines()));new=list(map(json.loads,(root/'lf_dataset'/(split+'.jsonl')).read_text().splitlines()));assert len(old)==len(ds) and len(new)==expected
 lookup={r['id']:(i,r) for i,r in enumerate(old)};indices=[];maximum=0
 for r in new:
  index,source=lookup[r['id']]
  assert r==source,'Subset row changed: '+r['id']
  cached=ds[index];ids=cached['input_ids'];labels=cached['labels'];active=[i for i in labels if i!=-100]
  assert len(ids)==len(labels)<cfg['cutoff_len'] and active and any(i==-100 for i in labels)
  assert tokenizer.decode(active,skip_special_tokens=True).strip()==r['conversations'][1]['value'].strip()
  for field in ['images','videos']:
   for path in r[field]:assert Path(path).is_file(),path
  maximum=max(maximum,len(ids));indices.append(index)
 derived[key]=ds.select(indices);report[split]=dict(rows=len(new),max_tokens=maximum,target_mismatch_count=0,source_sha256=hashlib.sha256((root/'lf_dataset'/(split+'.jsonl')).read_bytes()).hexdigest(),parent_indices_sha256=hashlib.sha256(json.dumps(indices).encode()).hexdigest())
assert set(r['source_id'] for r in map(json.loads,(root/'lf_dataset/train.jsonl').read_text().splitlines())).isdisjoint(set(r['source_id'] for r in map(json.loads,(root/'lf_dataset/val.jsonl').read_text().splitlines())))
DatasetDict(derived).save_to_disk(str(root/'lf_dataset/tokenized_v2'))
(root/'logs/lf_preflight_report.json').write_text(json.dumps(report,indent=2));(root/'logs/inherited_visual_audit.json').write_text(json.dumps(dict(parent_report=visual,parent_report_sha256=hashlib.sha256((proof/'lf_visual_audit_report.json').read_bytes()).hexdigest(),subset_rows=5882,exact_source_and_cache_match=True,no_media_or_order_changes=True),indent=2));print(json.dumps(report),flush=True)
