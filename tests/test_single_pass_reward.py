import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'grpo'))
from single_pass_reward import score_record


class SinglePassRewardTests(unittest.TestCase):
    def record(self, severe=0, general=0, valid=True):
        return dict(issues=[dict(severity='severe')] * severe + [dict(severity='general')] * general,
                    format_score=100 if valid else 0)

    def test_positive_raw_severe_reward_cannot_get_positive_advantage(self):
        result = score_record(self.record(severe=1))
        self.assertAlmostEqual(result['reward'], .2)
        self.assertFalse(result['positive_eligible'])

    def test_penalty_caps_and_floor(self):
        self.assertAlmostEqual(score_record(self.record(severe=3))['reward'], -.6)
        self.assertEqual(score_record(self.record(severe=3, general=9, valid=False))['reward'], -1.)

    def test_format_only_failure(self):
        result = score_record(self.record(valid=False))
        self.assertAlmostEqual(result['reward'], .7)
        self.assertFalse(result['positive_eligible'])

    def test_service_failure_has_no_numeric_training_reward(self):
        result = score_record(dict(judge_failed=True))
        self.assertIsNone(result['reward'])
        self.assertFalse(result['positive_eligible'])


if __name__ == '__main__':
    unittest.main()
