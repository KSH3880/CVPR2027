"""Measurement calibration: yaw invariance, rollover, and mass-independent margins."""
import math
import sys
import unittest
from pathlib import Path
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/multi_agent'))
from push_stability_common import tilt_degrees,static_envelope


class StabilityMetrics(unittest.TestCase):
    def test_small_tilt_with_arbitrary_yaw_and_scaled_quaternion(self):
        for pitch in (0.,.01,2.,5.,90.,179.):
            for yaw in (0.,47.,180.):
                p=math.radians(pitch)/2;y=math.radians(yaw)/2
                # qz(yaw) * qy(pitch), scalar-last; orientation scale is irrelevant.
                q=torch.tensor([[-math.sin(y)*math.sin(p),math.cos(y)*math.sin(p),
                                 math.sin(y)*math.cos(p),math.cos(y)*math.cos(p)]],dtype=torch.float64)*3
                self.assertAlmostEqual(tilt_degrees(q).item(),pitch,places=7)

    def test_weight_increases_slide_and_tip_forces_together(self):
        margins=[]
        for mass in (10.,15.,20.,25.,30.):
            r=static_envelope((1.5,1.2,1.5),mass,.65/1.5)
            self.assertAlmostEqual(r['sliding_reference_n'],.6*mass*9.81)
            margins.append(r['tipping_over_sliding'])
        self.assertTrue(all(abs(v-1.5384615384615385)<1e-12 for v in margins))

    def test_high_point_can_tip_before_sliding(self):
        low=static_envelope((1.2,1.2,1.5),20,.65/1.5)
        high=static_envelope((1.2,1.2,1.5),20,.75)
        self.assertGreater(low['tipping_over_sliding'],1)
        self.assertLess(high['tipping_over_sliding'],1)


if __name__=='__main__':unittest.main()
