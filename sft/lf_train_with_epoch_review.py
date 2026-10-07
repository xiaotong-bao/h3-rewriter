import json,sys,time
from pathlib import Path
sys.path.insert(0, '/work/epoch_inference_vendor')
from transformers import TrainerCallback
from llamafactory.train.tuner import run_exp
from common import processor,JOB
from epoch_benchmark import EpochBenchmark
class Record(TrainerCallback):
 def on_epoch_end(self,args,state,control,**kwargs):control.should_save=True
 def on_log(self,args,state,control,logs=None,**kwargs):
  if state.is_world_process_zero:
   (JOB/'progress.json').write_text(json.dumps(dict(step=state.global_step,max_steps=state.max_steps,epoch=state.epoch,time=time.time(),metrics=logs),indent=2))
config=json.loads(Path(sys.argv[1]).read_text())
run_exp(config,callbacks=[Record(),EpochBenchmark(processor())])
