"""Exactly two judge profiles: Astra model evaluation, Luna numeric reward."""
import os
from pathlib import Path
from judging_standard import ASTRA_REVISION, LUNA_REVISION
PROFILE = os.environ.get('H3_JUDGE_PROFILE', 'reward')
assert PROFILE in ('reward', 'evaluation'), 'Only the two published judge profiles are supported'
JUDGE_MODEL = 'gpt-6-luna' if PROFILE == 'reward' else 'gpt-6-astra'
JUDGE_REVISION = LUNA_REVISION if PROFILE == 'reward' else ASTRA_REVISION
JOB_NAME = os.environ.get('H3_GRPO_JOB', 'grpo_ep3_luna_2k_20261006')
JOB = Path('/work') / JOB_NAME
PORT = int(os.environ.get('H3_LUNA_PORT', '8797' if PROFILE == 'reward' else '8798'))
