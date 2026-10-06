"""Start the job-local Linux Codex judge supervisor; no SSH or Mac required."""
import json,os,pathlib,subprocess,sys
ROOT=pathlib.Path(__file__).resolve().parent
assert (ROOT/'luna_external_data_approval.json').exists(), 'Run bootstrap_runtime.py grpo first'
assert not (ROOT/'RUNTIME_STOP').exists(), 'Remove the explicit stop marker before restarting'
with (ROOT/'runtime_watchdog.log').open('a') as log:
 p=subprocess.Popen([sys.executable,'-u',str(ROOT/'runtime_watchdog.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'supervisor_pid':p.pid,'port':int(os.environ.get('H3_LUNA_PORT','8792')),'log':str(ROOT/'runtime_watchdog.log')}))
