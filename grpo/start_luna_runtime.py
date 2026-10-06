"""Start the job-local Linux Codex judge supervisor; no SSH or Mac required."""
import json,os,pathlib,subprocess,sys,time,urllib.request
from run_config import PORT
ROOT=pathlib.Path(__file__).resolve().parent
assert (ROOT/'luna_external_data_approval.json').exists(), 'Run bootstrap_runtime.py grpo first'
assert not (ROOT/'RUNTIME_STOP').exists(), 'Remove the explicit stop marker before restarting'
with (ROOT/'runtime_watchdog.log').open('a') as log:
 p=subprocess.Popen([sys.executable,'-u',str(ROOT/'runtime_watchdog.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'supervisor_pid':p.pid,'port':PORT,'log':str(ROOT/'runtime_watchdog.log')}))
deadline=time.monotonic()+30
while time.monotonic()<deadline:
 try:
  with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/',timeout=2) as r:health=json.load(r)
  from run_config import JUDGE_REVISION
  if health.get('ready') and health.get('revision')==JUDGE_REVISION and health.get('runtime')=='local_codex':break
 except Exception:pass
 time.sleep(.2)
else:raise SystemExit('Local gateway failed to become ready; inspect gateway.log')
