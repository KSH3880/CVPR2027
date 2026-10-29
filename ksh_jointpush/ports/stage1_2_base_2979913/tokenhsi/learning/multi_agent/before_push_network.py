"""Compatibility import: both five-task Stage2 tasks use the same builder."""
from learning.multi_agent.stage2_push_network import Stage2PushBuilder as BeforePushBuilder
from learning.multi_agent.stage2_push_network import transfer_stage1
