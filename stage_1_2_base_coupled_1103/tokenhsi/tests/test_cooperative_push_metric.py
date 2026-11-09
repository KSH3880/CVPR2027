"""Pure numpy regression tests; no simulation/GPU."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.cooperative_push_metric import surface_distance, score, sweep


def trace(frames=16):
    boxes = np.zeros((frames,1,2,13), dtype=np.float64)
    boxes[...,6] = 1
    boxes[...,0] = np.arange(frames)[:,None,None]*.1
    boxes[...,7] = 1
    hands = np.repeat(boxes[...,None,:3], 2, axis=-2)
    hands[...,0] += .55
    return dict(boxes=boxes, hands=hands, sizes=np.ones((1,2,3)),
                goals=np.full((1,2,3),10.), groups=np.zeros((1,2),int),
                templates=np.full((1,2),5), dt=np.array(.1), end_step=np.array([frames-1]))


class CoopTest(unittest.TestCase):
    def test_surface_inside_corner_rotation(self):
        box=np.array([0.,0,0,0,0,0,1])
        hands=np.array([[0,0,0],[.6,.6,.5],[.5,0,0]])
        np.testing.assert_allclose(surface_distance(hands,box,np.ones(3)), [.5,np.sqrt(.02),0])
        angle=.7
        rot=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
        box[3:7]=[0,0,np.sin(angle/2),np.cos(angle/2)]
        np.testing.assert_allclose(surface_distance(hands@rot.T,box,np.ones(3)), [.5,np.sqrt(.02),0], atol=1e-12)

    def test_displacement_all_hands_and_pause(self):
        d=trace()
        d['hands'][4,0,1,1,0] += 1  # only one hand leaves for one frame
        d['boxes'][5:,:, :,7:9]=0  # pauses remain in denominator
        r=score(d)['trials'][0]
        self.assertEqual((r['start_frame'],r['frames'],r['contact_frames']),(2,14,13))
        self.assertAlmostEqual(r['t_coop'],100*13/14)

    def test_strict_contact_threshold(self):
        d=trace()
        d['boxes'][...,0]=np.arange(16)[:,None,None]*.125
        d['hands'][...,0]=d['boxes'][...,None,0]+.625
        self.assertEqual(score(d,contact_threshold=.125)['trials'][0]['t_coop'],0)

    def test_speed_reset_and_backdate(self):
        d=trace(25)
        d['boxes'][5,...,7]=0
        r=score(d,'speed',speed_threshold=.5)['trials'][0]
        self.assertEqual(r['start_frame'],6)
        self.assertEqual(r['frames'],19)

    def test_confirmation_must_precede_success(self):
        d=trace()
        d['goals'][:]=[1.2,0,0]  # first success at frame 10 or 11 (float boundary)
        # Use a precise goal to guarantee first success at frame 10.
        d['goals'][:]=[1.19,0,0]
        r=score(d,'speed')['trials'][0]
        self.assertEqual(r['first_success_frame'],10)
        self.assertIsNone(r['start_frame'])
        self.assertEqual(r['t_coop'],0)
        d['goals'][:]=[1.29,0,0]
        r=score(d,'speed')['trials'][0]
        self.assertEqual((r['first_success_frame'],r['start_frame']), (11,1))

    def test_never_started_failure_retained(self):
        d=trace()
        d['boxes'][...,0]=0
        r=score(d)
        self.assertEqual(r['summary']['2']['trials'],1)
        self.assertEqual(r['summary']['2']['start_failures'],1)
        self.assertEqual(r['summary']['2']['t_coop_mean_percent'],0)

    def test_end_and_group_filter(self):
        d=trace()
        d['end_step'][:]=5
        self.assertEqual(score(d)['trials'][0]['frames'],4)
        d['templates'][0,1]=1
        self.assertEqual(score(d)['summary']['all']['trials'],0)

    def test_sweep_and_missing_raw(self):
        self.assertEqual(len(sweep(trace())),30)
        with self.assertRaisesRegex(ValueError,'missing'):
            score({'groups':np.zeros((1,2))})
        d=trace();d['boxes'][:,0,1,0]+=1
        with self.assertRaisesRegex(ValueError,'physical box'):
            score(d)

    def test_three_four_members_and_3d_goal(self):
        for members in (3,4):
            d=trace()
            for key in ('boxes','hands'):
                d[key]=np.repeat(d[key][:,:,:1],members,axis=2)
            for key in ('sizes','goals','groups','templates'):
                d[key]=np.repeat(d[key][:,:1],members,axis=1)
            d['goals'][:]=[.8,0,1]  # XY arrival alone cannot end measurement
            r=score(d)
            self.assertEqual(r['summary'][str(members)]['trials'],1)
            self.assertIsNone(r['trials'][0]['first_success_frame'])
            self.assertEqual(r['trials'][0]['t_coop'],100)


if __name__ == '__main__':
    unittest.main()
