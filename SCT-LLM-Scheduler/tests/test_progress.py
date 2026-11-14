import time
import unittest
from sct_scheduler.progress import StageETA
class ProgressTests(unittest.TestCase):
    def test_reuse_excluded(self):
        now=time.time();start=now-100
        s=dict(stage='api',total=100,api_done=60,api_reused=50,started_at=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(start)))
        StageETA().annotate(s,now)
        self.assertAlmostEqual(s['eta_seconds'],400,delta=4)
    def test_stage_change_resets(self):
        eta=StageETA();s=dict(stage='eval',verified=[],total_evaluations=3)
        eta.annotate(s,100);self.assertIsNone(s['eta_seconds'])
        s['verified']=['one'];eta.annotate(s,130);self.assertEqual(s['eta_seconds'],60)
        s['stage']='metrics';eta.annotate(s,140);self.assertIsNone(s['eta_seconds'])
        s['completed']=True;eta.annotate(s,150);self.assertEqual(s['eta_seconds'],0)
