"""Shared configuration for the isolated, Linux-local v5 rerun."""
import os
from pathlib import Path
JUDGE_REVISION = 'ep3-v2-luna-state-v5'
JOB_NAME = os.environ.get('H3_GRPO_JOB', 'grpo_ep3_luna_v5_20261006')
JOB = Path('/work') / JOB_NAME
PORT = int(os.environ.get('H3_LUNA_PORT', '8794'))
