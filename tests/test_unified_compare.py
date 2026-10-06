import importlib.util,json,tempfile,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('unified_compare',Path(__file__).resolve().parents[1]/'review/unified_compare.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
class UnifiedCompareTests(unittest.TestCase):
 def rows(self):
  return [{'id':'case1','label':label,'format':{'fields':True},'audit':{'severity_counts':{'severe':0},'maximum_severity':'none','reward':1.,'unrequested_music':False,'unrequested_dialogue':False}} for label in module.LABELS]
 def test_failed_judge_excludes_case_from_all_semantic_columns(self):
  rows=self.rows();rows[0]['audit']={'judge_failed':True};rows[1]['format']={'fields':False}
  with tempfile.TemporaryDirectory() as tmp:s=module.summarize(rows,Path(tmp),'test')
  self.assertEqual(s['common_valid'],0)
  for v in s['models'].values():self.assertEqual(v['valid_common'],0);self.assertIsNone(v['mean_reward'])
  self.assertEqual(s['models']['ep3']['format_failures'],1)
 def test_partial_coverage_cannot_be_complete(self):
  with tempfile.TemporaryDirectory() as tmp:s=module.summarize(self.rows(),Path(tmp),'test')
  self.assertEqual(s['common_valid'],1);self.assertFalse(s['complete'])
 def test_same_case_uses_same_denominator_for_every_model(self):
  rows=self.rows();rows[0]['audit'].update(severity_counts={'severe':1},maximum_severity='severe',reward=.2)
  with tempfile.TemporaryDirectory() as tmp:s=module.summarize(rows,Path(tmp),'test')
  self.assertEqual(s['models']['fal']['severe'],1)
  self.assertTrue(all(v['valid_common']==1 for v in s['models'].values()))
if __name__=='__main__':unittest.main()
