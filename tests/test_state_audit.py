import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'grpo'))
from reward_policy import validate_and_score, positive_eligible

class StateAuditTests(unittest.TestCase):
    def verdict(self,status='contradicted'):
        return {'checklist':[{'requirement_id':1,'status':'preserved','severity':'none','evidence_ids':[1],'reason':'Earlier text appears correct'}],
                'v2_issues':[],'unrequested_music':False,'unrequested_dialogue':False,
                'state_audit':[{'source_id':1,'expected_initial':'Fog','expected_transition':'Fades','expected_final':'Clear glass','actual_final':'Still foggy',
                                'status':status,'evidence_ids':[1],'reason':'Last sentence reverses evaporation'}]}
    def score(self,v):
        return validate_and_score('Fog fades from the window.','Fog evaporates, leaving the glass fog-obscured.',
            [{'id':1,'text':'Fog fades','critical':True,'source_ids':[1]}],v,require_state_audit=True)
    def test_false_preserved_first_pass_is_overridden(self):
        v=self.score(self.verdict());self.assertEqual(v['severity_counts']['severe'],1)
        self.assertAlmostEqual(v['reward'],.2);self.assertFalse(positive_eligible(v,True))
    def test_missing_audit_fails_closed(self):
        v=self.verdict();del v['state_audit']
        with self.assertRaises(AssertionError):self.score(v)
    def test_missing_source_sentence_fails_closed(self):
        v=self.verdict();v['state_audit']=[]
        with self.assertRaises(AssertionError):self.score(v)
    def test_state_constraint_cannot_be_not_applicable(self):
        with self.assertRaises(AssertionError):self.score(self.verdict('not_applicable'))
    def test_uncertainty_blocks_positive(self):
        v=self.score(self.verdict('uncertain'));self.assertFalse(positive_eligible(v,True))
    def test_no_duplicate_severe_penalty(self):
        v=self.verdict();v['checklist'][0].update(status='contradicted',severity='severe')
        v=self.score(v);self.assertEqual(v['severity_counts']['severe'],1)
