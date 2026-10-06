import copy,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'grpo'))
from reward_policy import validate_and_score,positive_eligible,gate_advantage

class RewardPolicyTests(unittest.TestCase):
    def setUp(self):
        self.source='The cube rolls left. It stops at the cone.'
        self.rewrite=self.source
        self.reqs=[{'id':1,'text':'The cube rolls left','critical':True,'source_ids':[1]},
                   {'id':2,'text':'The cube stops at the cone','critical':True,'source_ids':[2]}]
        self.verdict={'checklist':[{'requirement_id':i,'status':'preserved','severity':'none','evidence_ids':[i],'reason':'Directly stated'} for i in (1,2)],
                      'v2_issues':[],'unrequested_music':False,'unrequested_dialogue':False}
    def score(self,v):return validate_and_score(self.source,self.rewrite,self.reqs,v)
    def test_faithful_checklist_passes(self):
        v=self.score(copy.deepcopy(self.verdict));self.assertEqual(v['reward'],1);self.assertTrue(positive_eligible(v,True))
    def test_missing_requirement_rejected(self):
        v=copy.deepcopy(self.verdict);v['checklist'].pop()
        with self.assertRaises(AssertionError):self.score(v)
    def test_duplicate_requirement_rejected(self):
        v=copy.deepcopy(self.verdict);v['checklist'][1]['requirement_id']=1
        with self.assertRaises(AssertionError):self.score(v)
    def test_music_flag_without_issue_rejected(self):
        v=copy.deepcopy(self.verdict);v['unrequested_music']=True
        with self.assertRaises(AssertionError):self.score(v)
    def test_music_issue_without_flag_rejected(self):
        v=copy.deepcopy(self.verdict);v['v2_issues']=[{'category':'v2_music','covered_requirement_id':0,'severity':'general','evidence_ids':[1],'reason':'Unrequested score'}]
        with self.assertRaises(AssertionError):self.score(v)
    def test_critical_downgrade_rejected(self):
        v=copy.deepcopy(self.verdict);v['checklist'][1].update(status='contradicted',severity='general')
        with self.assertRaises(AssertionError):self.score(v)
    def test_critical_wrong_ending_blocks_positive(self):
        v=copy.deepcopy(self.verdict);v['checklist'][1].update(status='contradicted',severity='severe')
        self.assertFalse(positive_eligible(self.score(v),True))
    def test_bad_format_never_gets_positive_advantage(self):
        v=self.score(copy.deepcopy(self.verdict));self.assertFalse(positive_eligible(v,False))
        self.assertEqual(gate_advantage(.7-.675,False),0)
        self.assertEqual(gate_advantage(-.5,False),-.5)
    def test_failed_judge_has_no_positive_or_negative_policy_signal(self):
        self.assertFalse(positive_eligible({'judge_failed':True},True))
        self.assertEqual(gate_advantage(-10,False,True),0)
    def test_music_is_penalized_once_and_blocks_positive(self):
        v=copy.deepcopy(self.verdict);v['unrequested_music']=True
        v['v2_issues']=[{'category':'v2_music','covered_requirement_id':0,'severity':'general','evidence_ids':[1],'reason':'Unrequested score'}]
        v=self.score(v);self.assertAlmostEqual(v['reward'],.9);self.assertFalse(positive_eligible(v,True))
    def test_explicit_music_prohibition_is_not_double_counted(self):
        req=[{'id':1,'text':'No music','category':'music','critical':True,'source_ids':[1]}]
        v={'checklist':[{'requirement_id':1,'status':'contradicted','severity':'severe','evidence_ids':[1],'reason':'Music added'}],
           'v2_issues':[{'category':'v2_music','covered_requirement_id':1,'severity':'severe','evidence_ids':[1],'reason':'Music added'}],
           'unrequested_music':True,'unrequested_dialogue':False}
        v=validate_and_score('No music is allowed.','A piano melody plays.',req,v)
        self.assertAlmostEqual(v['reward'],.2);self.assertEqual(len(v['items']),1)
if __name__=='__main__':unittest.main()
