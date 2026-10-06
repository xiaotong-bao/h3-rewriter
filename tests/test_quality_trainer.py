import sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'grpo'))
import torch
from accelerate import PartialState
import train_grpo as t

class QualityTrainerTests(unittest.TestCase):
    def records(self):return [dict(eligible=False,judge_failed=False,token_ids=[1,2],original='A',reward=.7,format_valid=False),
         dict(eligible=True,judge_failed=False,token_ids=[3,4],original='B',reward=.9,format_valid=True),
         dict(eligible=False,judge_failed=True,token_ids=[5,6],original='C',reward=None,format_valid=True)]
    def output(self):return {'advantages':torch.tensor([.25,-.5,.2]),'completion_ids':torch.tensor([[1,2],[3,4],[5,6]]),'completion_mask':torch.ones(3,2,dtype=torch.int64)}
    def test_gate_is_applied_to_actual_trainer_output(self):
        obj=t.QualityGatedGRPOTrainer.__new__(t.QualityGatedGRPOTrainer);obj.state=SimpleNamespace(global_step=0)
        with tempfile.TemporaryDirectory() as d,patch.object(t,'ROOT',Path(d)),patch.object(t,'_LATEST_RECORDS',self.records()),patch.object(t.GRPOTrainer,'_generate_and_score_completions',return_value=self.output()):
            out=obj._generate_and_score_completions([])
            self.assertEqual(out['advantages'].tolist(),[0,-.5,0])
            self.assertEqual(out['completion_mask'][2].sum().item(),0)
            self.assertEqual(out['completion_mask'][0].sum().item(),2)
    def test_wrong_token_alignment_fails_closed(self):
        obj=t.QualityGatedGRPOTrainer.__new__(t.QualityGatedGRPOTrainer);obj.state=SimpleNamespace(global_step=0)
        out=self.output();out['completion_ids'][0,0]=9
        with tempfile.TemporaryDirectory() as d,patch.object(t,'ROOT',Path(d)),patch.object(t,'_LATEST_RECORDS',self.records()),patch.object(t.GRPOTrainer,'_generate_and_score_completions',return_value=out):
            with self.assertRaises(AssertionError):obj._generate_and_score_completions([])
    def test_fixed_trl_maps_none_reward_to_unscorable_nan(self):
        PartialState(cpu=True)
        obj=t.QualityGatedGRPOTrainer.__new__(t.QualityGatedGRPOTrainer)
        obj.accelerator=SimpleNamespace(device=torch.device('cpu'),is_main_process=False);obj.state=SimpleNamespace(global_step=0);obj.environments=None
        obj.reward_funcs=[lambda **kw:[1.,None]];obj.reward_processing_classes=[None];obj.reward_func_names=['test']
        obj.args=SimpleNamespace(report_to=[]);obj._log_completion_extra=lambda **kw:None;obj._log_metric=lambda *args:None
        r=obj._calculate_rewards([{'original':'A'},{'original':'B'}],['A','B'],['a','b'],[[1],[2]])
        self.assertEqual(r[0,0].item(),1);self.assertTrue(torch.isnan(r[1,0]))
if __name__=='__main__':unittest.main()
