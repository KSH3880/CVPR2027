"""CPU geometry checks for full-body 360-degree collision-free PUSH starts."""
import math
import torch
from utils.door_reference import rotate
from utils.push_random_spawn import bounds_clearance, place_at_surface_gap, random_approach, unsafe_initial_rows


class Geometry:
    margin = .02

    def bounds(self, body, rotations, boxes):
        inv = boxes[:, 3:7].clone()
        inv[:, :3] *= -1
        centers = rotate(inv[:, None].expand(-1, body.shape[1], -1), body-boxes[:, None, :3])
        # Asymmetric per-body collider bounds (arms and feet, not just root).
        radius = body.new_tensor([.2, .15, .12])[None, :, None]
        return centers-radius, centers+radius

    def floor_shift(self, body, rotations):
        return torch.zeros(len(body), device=body.device)

    def invalid(self, body, rotations, boxes, sizes):
        lo, hi = self.bounds(body, rotations, boxes)
        return ((hi+self.margin > -sizes[:, None]/2)&(lo-self.margin < sizes[:, None]/2)).all(-1).any(-1)


def fixture(n=128):
    body = torch.tensor([[0., 0., 0.], [.8, .1, .4], [-.3, -.2, -.9]])[None].repeat(n,1,1)
    rotations = torch.zeros(n,3,4)
    rotations[...,3] = 1
    boxes = torch.zeros(n,13)
    boxes[...,2] = .65
    boxes[...,6] = 1
    sizes = torch.tensor([1.3,2.,1.3])[None].repeat(n,1)
    return body,rotations,boxes,sizes,torch.ones(n)


def test_full_body_surface_gap_independent_of_box_yaw_and_width():
    n=256
    body,rot,boxes,sizes,z=fixture(n)
    yaw=torch.linspace(0,2*math.pi,n)
    boxes[:,5],boxes[:,6]=(yaw/2).sin(),(yaw/2).cos()
    boxes[:,:2]=torch.randn(n,2)*100
    sizes[:,0]=torch.linspace(1.2,1.5,n)
    bearing=torch.linspace(0,2*math.pi,n)
    requested=torch.linspace(1.,5.,n)
    placed=place_at_surface_gap(Geometry(),body,rot,boxes,sizes,z,bearing,requested)
    lo,hi=Geometry().bounds(placed,rot,boxes)
    torch.testing.assert_close(bounds_clearance(lo,hi,sizes),requested,atol=1e-4,rtol=0)
    assert not Geometry().invalid(placed,rot,boxes,sizes).any()
    # Rigid placement leaves every pair of reference body offsets unchanged.
    torch.testing.assert_close(placed-placed[:,:1],body-body[:,:1],atol=2e-5,rtol=0)


def test_random_bearings_cover_all_sides_with_surface_gap_1_to_5():
    torch.manual_seed(4)
    body,rot,boxes,sizes,z=fixture(4096)
    placed=random_approach(Geometry(),body,rot,boxes,sizes,z,
                           {'loco_surface_clearance_range_m':[1.,5.]})
    delta=placed[:,0,:2]-boxes[:,:2]
    quadrant=(delta[:,0]>0).long()*2+(delta[:,1]>0).long()
    counts=torch.bincount(quadrant,minlength=4)
    assert (counts>850).all()
    lo,hi=Geometry().bounds(placed,rot,boxes)
    gap=bounds_clearance(lo,hi,sizes)
    assert gap.min() >= .9999 and gap.max() <= 5.0001


def test_all_objects_are_checked_and_foot_arm_overlap_is_rejected():
    body,rot,own,sizes,z=fixture(4)
    placed=place_at_surface_gap(Geometry(),body,rot,own,sizes,z,torch.zeros(4),torch.ones(4))
    pose=torch.cat((placed,rot,torch.zeros(4,3,6)),-1)
    boxes=own[:,None].repeat(1,4,1)
    boxes[:,1:,:2]+=30
    all_sizes=sizes[:,None].repeat(1,4,1)
    assert not unsafe_initial_rows(Geometry(),pose,boxes,all_sizes,torch.full((4,),.2)).any()
    # The fourth unassigned box intersects a non-root collider.
    boxes[1,3,:3]=placed[1,2]
    result=unsafe_initial_rows(Geometry(),pose,boxes,all_sizes,torch.full((4,),.2))
    assert result.tolist() == [False,True,False,False]


def test_gap_uses_all_colliders_instead_of_human_center():
    lo=torch.tensor([[[2.,0.,0.],[.6,0.,0.]]])
    hi=lo+.1
    sizes=torch.ones(1,3)
    torch.testing.assert_close(bounds_clearance(lo,hi,sizes),torch.tensor([.1]))
