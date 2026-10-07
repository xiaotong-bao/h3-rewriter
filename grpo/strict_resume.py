"""Retain optimizer moments and RNG, but extend the completed LR schedule."""
import hashlib,json
from pathlib import Path
import torch
from transformers import TrainerCallback

def optimizer_moment_digest(state):
    h=hashlib.sha256()
    for key,values in sorted(state['state'].items()):
        h.update(str(key).encode())
        for name,value in sorted(values.items()):
            h.update(name.encode())
            if isinstance(value,torch.Tensor):h.update(value.detach().cpu().contiguous().numpy().tobytes())
            else:h.update(repr(value).encode())
    return h.hexdigest()

def continuation_factor(step,start,end):
    assert end>start
    return max(0.,min(1.,(end-step)/(end-start)))

class ResumeSchedule(TrainerCallback):
    def __init__(self,trainer,checkpoint,rate,policy_hash,reference_hash):
        self.trainer=trainer;self.checkpoint=Path(checkpoint);self.rate=rate;self.policy_hash=policy_hash;self.reference_hash=reference_hash
    def on_train_begin(self,args,state,control,**kwargs):
        from train_userlike_grpo import adapter_hash
        saved=json.loads((self.checkpoint/'trainer_state.json').read_text());start=saved['global_step']
        assert state.global_step==start
        state.save_steps = args.save_steps
        state.logging_steps = args.logging_steps
        assert adapter_hash(self.trainer.model,'default')==self.policy_hash
        assert adapter_hash(self.trainer.model,'ref')==self.reference_hash
        old=torch.load(self.checkpoint/'optimizer.pt',map_location='cpu',weights_only=False)
        expected=optimizer_moment_digest(old);actual=optimizer_moment_digest(self.trainer.optimizer.state_dict())
        assert actual==expected,'Optimizer moments/steps were not strictly restored'
        scheduler=self.trainer.lr_scheduler
        assert scheduler.last_epoch==start,'Scheduler counter not restored'
        scheduler.base_lrs=[self.rate]*len(scheduler.base_lrs)
        scheduler.lr_lambdas=[lambda step:continuation_factor(step,start,args.max_steps)]*len(scheduler.base_lrs)
        scheduler._last_lr=[self.rate]*len(scheduler.base_lrs)
        for group in self.trainer.optimizer.param_groups:group['lr']=self.rate;group['initial_lr']=self.rate
        if self.trainer.is_world_process_zero():
            Path(args.output_dir,'resume_receipt.json').write_text(json.dumps(dict(checkpoint=str(self.checkpoint),global_step=start,target_step=args.max_steps,optimizer_moments_sha256=actual,optimizer_restored=True,policy_hash=self.policy_hash,original_reference_hash=self.reference_hash,learning_rate=self.rate,schedule='linear from resumed global step to new target',rng_files=[f.name for f in self.checkpoint.glob('rng_state_*.pth')],new_dataset_starts_from_beginning=True),indent=2)+'\n')
        print('STRICT_RESUME_VERIFIED',start,'LR',self.rate,'TARGET',args.max_steps,flush=True)
