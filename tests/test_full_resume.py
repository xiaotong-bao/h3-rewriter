import unittest
import torch
from grpo.single_pass_reward import format_for_task
from grpo.strict_resume import optimizer_moment_digest,continuation_factor
from grpo.video_grpo import install_video_support,video_trainer_class
import trl.trainer.grpo_trainer as runtime

class FullResumeTests(unittest.TestCase):
 def test_reference_task_requires_six_fields(self):
  keys=['subject_definitions','summary','retention_analysis','detailed_description','overall_soundscape','non_diegetic_music']
  self.assertTrue(format_for_task('\n'.join(k+': value' for k in keys),'ref2va'))
  self.assertFalse(format_for_task('integrated_multimodal_description: x\noverall_soundscape: x\nnon_diegetic_music: x','ref2va'))
 def test_schedule_restarts_at_resume_counter(self):
  self.assertEqual(continuation_factor(500,500,5330),1)
  self.assertAlmostEqual(continuation_factor(2915,500,5330),.5)
  self.assertEqual(continuation_factor(5330,500,5330),0)
 def test_lr_change_does_not_change_moment_digest(self):
  s={'state':{0:{'step':torch.tensor(500.),'exp_avg':torch.tensor([1.,2.]),'exp_avg_sq':torch.tensor([3.,4.])}},'param_groups':[{'lr':0.}]}
  old=optimizer_moment_digest(s);s['param_groups'][0]['lr']=2e-6;self.assertEqual(optimizer_moment_digest(s),old)
  s['state'][0]['exp_avg'][0]=5.;self.assertNotEqual(optimizer_moment_digest(s),old)
 def test_modality_ids_match_scored_sequence(self):
  class Parent:
   def _get_per_token_logps_and_entropies(self,model,input_ids,*args,**kwargs):return kwargs
  class Tokenizer:
   def convert_tokens_to_ids(self,token):return {'<|image_pad|>':11,'<|video_pad|>':12}[token]
  trainer=video_trainer_class(Parent)();trainer._tokenizer=Tokenizer()
  ids=torch.tensor([[1,11,12,3,4]])
  out=trainer._get_per_token_logps_and_entropies(None,ids,mm_token_type_ids=torch.zeros(1,2))
  self.assertTrue(torch.equal(out['mm_token_type_ids'],torch.tensor([[0,1,2,0,0]])))
 def test_video_buffer_split_and_reassemble(self):
  install_video_support()
  data={'num_videos':[1,0,2],'video_grid_thw':torch.tensor([[2,2,2],[2,1,2],[2,1,2]]),'pixel_values_videos':torch.arange(16).reshape(16,1)}
  split=runtime.split_pixel_values_by_grid(data)
  self.assertEqual([len(t) for t in split['pixel_values_videos']],[8,0,8])
  combined=runtime.unsplit_pixel_values_by_grid(split)
  self.assertTrue(torch.equal(combined['pixel_values_videos'],data['pixel_values_videos']))
if __name__=='__main__':unittest.main()
