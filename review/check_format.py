"""Mechanical format audit of any t2va/i2va set, without normalizing raw output."""
import argparse,hashlib,json,pathlib,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'grpo'))
from format_rules import check_format
p=argparse.ArgumentParser();p.add_argument('--input',required=True,type=pathlib.Path);p.add_argument('--sources',required=True,type=pathlib.Path);p.add_argument('--output',required=True,type=pathlib.Path)
a=p.parse_args()
rows=[json.loads(s) for s in a.input.read_text().splitlines()]
sources={r['id']:r for r in map(json.loads,a.sources.read_text().splitlines())}
assert len(rows)==len({r['id'] for r in rows})==len(sources)
assert {r['id'] for r in rows}==set(sources)
records=[]
for row in rows:
 source=sources[row['id']];assert source['task'] in ['t2va','i2va']
 if 'original_prompt' in row and 'original_prompt' in source:assert row['original_prompt'].strip()==source['original_prompt'].strip()
 checks=check_format(row['rewrite'],source['task'],source['duration_s'])
 records.append({'id':row['id'],'checks':checks,'pass':all(checks.values()),'failed_checks':[k for k,v in checks.items() if not v]})
report={'input_sha256':hashlib.sha256(a.input.read_bytes()).hexdigest(),'sources_sha256':hashlib.sha256(a.sources.read_bytes()).hexdigest(),'rows':len(rows),'passed':sum(r['pass'] for r in records),'records':records,'scope':'Mechanical structure only; semantics, image fidelity and H3 encoder acceptance are separate.'}
a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))
