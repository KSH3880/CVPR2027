import math
import unittest
import torch
from carry_planner.proximity_metrics import proximity_distances, ProximityTracker, box_box_distance, rotation_matrix


class ProximityTest(unittest.TestCase):
    def boxes(self, x):
        b=torch.zeros(1,2,13);b[...,6]=1;b[0,1,0]=x
        return b,torch.ones(1,2,3)

    def test_axis_distance_overlap_and_rotated_overlap(self):
        for x,expected in [(2.,1.),(1.2,.2),(1.,0.),(.5,0.)]:
            b,size=self.boxes(x)
            got=box_box_distance(b[...,:3],rotation_matrix(b[...,3:7]),size/2)
            self.assertAlmostEqual(got.item(),expected,places=5)
        b,size=self.boxes(0)
        b[0,1,5]=math.sin(math.pi/8);b[0,1,6]=math.cos(math.pi/8)
        self.assertEqual(box_box_distance(b[...,:3],rotation_matrix(b[...,3:7]),size/2).item(),0)

    def test_other_box_only_and_rotated_surface(self):
        b,size=self.boxes(4)
        body=torch.tensor([[[[0.,0.,0.]],[[4.,0.,0.]]]])
        d=proximity_distances(body,b,size)
        self.assertAlmostEqual(d['agent_box'].item(),3.5)
        self.assertAlmostEqual(d['agent_agent'].item(),4.)
        b[0,1,5]=math.sin(math.pi/8);b[0,1,6]=math.cos(math.pi/8)
        self.assertAlmostEqual(proximity_distances(body,b,size)['agent_box'].item(),4-math.sqrt(.5),places=5)

    def test_total_union_threshold_and_reset(self):
        t=ProximityTracker(2,'cpu',.3)
        t.update(dict(agent_agent=torch.tensor([.2,.3]),agent_box=torch.tensor([.1,.4]),box_box=torch.tensor([.4,.1])))
        self.assertEqual(t.steps.tolist(),[[1,1,0,1],[0,0,1,1]])
        t.reset(torch.tensor([0]))
        self.assertEqual(t.steps.tolist(),[[0,0,0,0],[0,0,1,1]])
