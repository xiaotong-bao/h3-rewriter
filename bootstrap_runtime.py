"""Write local run receipts; never publish generated files or credentials."""
import argparse,hashlib,json,pathlib
parser=argparse.ArgumentParser()
parser.add_argument('mode',choices=['sft','grpo']);parser.add_argument('--root',type=pathlib.Path,default=pathlib.Path('/work'))
parser.add_argument('--job',default='grpo_ep3_luna_v4_20261006')
parser.add_argument('--include-calibration-sources',action='store_true')
args=parser.parse_args();root=args.root
if args.mode=='sft':
 job=root/'trl_sft_official_v2_20261006';job.mkdir(exist_ok=True)
 digest=hashlib.sha256((root/'models/Qwen3.5-9B/chat_template.jinja').read_bytes()).hexdigest()
 expected='a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715'
 assert digest==expected,'Template differs from verified official revision'
 payload={'local_sha256':digest,'official_sha256':expected,'official_revision':'c202236235762e1c871ad0ccb60c8ee5ba337b9a'}
 (job/'official_template_receipt.json').write_text(json.dumps(payload,indent=2))
else:
 job=root/args.job
 inputs=[json.loads(s) for s in (job/'pilot_inputs.jsonl').read_text().splitlines()]
 benchmark=[json.loads(s) for s in (root/'trl_sft_official_v2_20261006/epoch_benchmarks/step906/results.jsonl').read_text().splitlines()]
 sources=[r['original'] for r in inputs]+[r['original_prompt'] for r in benchmark]
 if args.include_calibration_sources:
  import sys
  sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent/'grpo'))
  from calibrate_judge import cases
  sources += [r['original'] for r in cases()]
 allow={'sha256':sorted({hashlib.sha256(s.strip().encode()).hexdigest() for s in sources})}
 (job/'luna_approved_sources.json').write_text(json.dumps(allow))
 (job/'luna_external_data_approval.json').write_text(json.dumps({'approved':True,'scope':'User-authorized Linux-local v4 rerun: 256 training prompts, 101 evaluation prompts'+(' and 5 synthetic regression sources' if args.include_calibration_sources else '')}))
 (job/'direct_grpo_authorization.json').write_text(json.dumps({'approved':True,'judge_revision':'ep3-v2-luna-checklist-v4','full_256_audit_gate_waived_by_user':True,'steps':64,'judge_concurrency':16}))
print('Runtime receipts written locally for',args.mode)
