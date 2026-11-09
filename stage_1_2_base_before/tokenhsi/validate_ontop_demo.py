"""Independently recheck saved demo geometry and timing on CPU."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from utils.evaluation_protocol import spatial_success
from learning.multi_agent.ontop_demo import qualifying_wait, holding_through_gate, center_above_support
from env.tasks.multi_agent.edge_ontop_reward import vertical_extent


def validate(directory):
    directory = Path(directory)
    meta = json.loads((directory/'metadata.json').read_text())
    x = np.load(directory/'trajectory.npz')
    m, leader, follower = meta['num_agents'],meta['leader'],meta['follower']
    n = len(x['roots'])
    order = x['logical_box_order']
    packet = torch.tensor(x['packet']).unsqueeze(0).expand(n,-1,-1)
    boxes = x['roots'][:,m+order]
    current = spatial_success(packet,torch.tensor(x['roots'][:,:m,:3]),
        torch.tensor(boxes),torch.tensor(x['sizes_physical'][order]).unsqueeze(0).expand(n,-1,-1),
        torch.tensor(x['goals']).unsqueeze(0).expand(n,-1,-1))[...,1].numpy()
    spatial = current[:,leader] & current[:,follower]
    assert np.array_equal(spatial,x['spatial']), 'CPU spatial scoring differs from recorded GPU scoring'
    distance = np.linalg.norm(x['roots'][:,follower,:2]-x['pair_goal'][:2],axis=-1)
    speed = np.linalg.norm(x['roots'][:,follower,7:9],axis=-1)
    assert np.allclose(distance,x['distance'],atol=1e-5)
    assert np.allclose(speed,x['speed'],atol=1e-5)
    radius = meta.get('search_radius',2.)
    carry_mask = x['holding'] if meta.get('require_carry_while_waiting',True) else np.ones(n,dtype=bool)
    if meta.get('stable_carry_while_waiting',False):
        payload_id=int(order[int(x['packet'][follower,3])-m])
        payload_speed=np.linalg.norm(x['roots'][:,m+payload_id,7:10],axis=-1)
        assert np.allclose(payload_speed,x['payload_speed'],atol=1e-5)
        carry_mask=carry_mask & holding_through_gate(x['holding'],x['gate'])
        if not meta.get('wait_extent_m'):
            carry_mask=carry_mask & (payload_speed<=.25)
    if meta.get('require_walk_before_wait',False):
        start_xy=x['initial_roots'][follower,:2]
        displacement=np.linalg.norm(x['roots'][:,follower,:2]-start_xy,axis=-1)
        initial_distance=np.linalg.norm(start_xy-x['pair_goal'][:2])
        carry_mask=carry_mask & (displacement>=1.) & ((initial_distance-distance)>=.75)
    wait_options=dict(positions=x['roots'][:,follower,:2],max_extent=meta.get('wait_extent_m'))
    wait = qualifying_wait(distance,speed,carry_mask,x['gate'],x['valid'],meta['dt'],seconds=meta.get('wait_required_seconds',1.), **wait_options)
    if wait is None:
        wait = qualifying_wait(distance,speed,carry_mask,x['gate'],x['valid'],meta['dt'],radius=radius,seconds=meta.get('wait_required_seconds',1.), **wait_options)
    assert wait is not None
    first_gate = int(np.flatnonzero(x['gate'])[0])
    assert first_gate+1 == meta['first_predecessor_success_step']
    if meta.get('stable_carry_while_waiting',False):
        assert x['holding'][wait[0]:first_gate+1].all(), 'Payload released before gate'
    end = meta['success_step']
    span = int(np.ceil(.5/meta['dt']-1e-6))
    if meta.get('success_mode')=='center_above':
        payload_idx=int(order[int(x['packet'][follower,3])-m])
        support_idx=int(order[int(x['packet'][follower,4])-m])
        broad=center_above_support(torch.tensor(x['roots'][:,m+payload_idx]),torch.tensor(x['roots'][:,m+support_idx]),torch.tensor(x['sizes_physical'][support_idx])).numpy()
        assert np.array_equal(broad,x['center_above'])
        assert (broad[end-span:end] & x['gate'][end-span:end] & current[end-span:end,leader] & x['valid'][end-span:end]).all()
        assert not current[:first_gate,follower].any()
    else:
        assert (spatial[end-span:end] & x['success'][end-span:end] & x['valid'][end-span:end]).all()
    climbing = meta.get('task_name') == 'CLIMB'
    sitting = meta.get('task_name') == 'SIT'
    payload = int(order[int(x['packet'][follower,4 if (climbing or sitting) else 3])-m])
    support = int(order[int(x['packet'][follower,4])-m])
    p = x['roots'][:,m+payload]
    s = x['roots'][:,m+support]
    if 'hand_body_ids' in meta:
        hands = x['bodies'][:,follower][:,meta['hand_body_ids'],:3].mean(1)
        hand_distance = np.linalg.norm(hands-p[:,:3],axis=-1)
        extent = vertical_extent(torch.tensor(p[:,3:7]),torch.tensor(x['sizes_physical'][payload]/2)).numpy()
        bottom = p[:,2]-extent
        carrying = (bottom>.15) & (hand_distance<.55)
        assert np.array_equal(carrying,x['holding']), 'CPU carrying geometry differs'
        assert np.allclose(bottom,x['payload_bottom'],atol=1e-5)
        assert np.allclose(hand_distance,x['hand_distance'],atol=1e-5)
    # Report the actual payload motion and the native hand-proximity HOLDING signal,
    # without calling a geometric ON_TOP label proof of physical release.
    result = dict(cpu_spatial_and_timing_verified=True, success_mode=meta.get('success_mode','strict'),
        native_and_strict_success_at_selected_end=bool((spatial[end-span:end] & x['success'][end-span:end]).all()), wait_radius_m=radius,
        first_gate_seconds=(first_gate+1)*meta['dt'],
        success_seconds=end*meta['dt'],
        wait_start_seconds=(wait[0]+1)*meta['dt'],wait_end_seconds=(wait[1]+1)*meta['dt'],
        wait_max_distance_m=float(distance[wait[0]:wait[1]+1].max()),
        wait_max_speed_mps=float(speed[wait[0]:wait[1]+1].max()),
        payload_speed_final_mean=float(np.linalg.norm(p[end-span:end,7:10],axis=-1).mean()),
        carrying_at_success_fraction=float(x['holding'][end-span:end].mean()),
        native_holding_at_success_fraction=float(x['native_holding'][end-span:end].mean()) if 'native_holding' in x else None,
        distance_at_gate=float(distance[first_gate]),speed_at_gate=float(speed[first_gate]),
        distance_max_wait_to_gate=float(distance[wait[0]:first_gate+1].max()),
        payload_center_z_at_wait=float(p[wait[1],2]),payload_center_z_at_success=float(p[end-1,2]),
        support_center_z_at_success=float(s[end-1,2]),
        support_height=float(x['sizes_physical'][support,2]),
        payload_height=float(x['sizes_physical'][payload,2]))
    xy=x['roots'][wait[0]:wait[1]+1,follower,:2]
    result['wait_xy_diameter_m']=float(np.linalg.norm(xy[:,None]-xy[None,:],axis=-1).max())
    if climbing or sitting:
        assert not current[:first_gate,follower].any(), 'Follower climbed before prerequisite completion'
        result['no_climb_success_before_gate'] = True
        result['first_climb_spatial_seconds'] = float((np.flatnonzero(current[:,follower])[0]+1)*meta['dt'])
        result['follower_root_z_at_success'] = float(x['roots'][end-1,follower,2])
        result['task_name'] = meta['task_name']
        if sitting:
            result['no_sit_success_before_gate']=result.pop('no_climb_success_before_gate')
            result['first_sit_spatial_seconds']=result.pop('first_climb_spatial_seconds')
    if meta.get('require_walk_before_wait',False):
        result['walk_displacement_m']=float(displacement[wait[0]])
        result['approach_distance_m']=float(initial_distance-distance[wait[0]])
    (directory/'validation.json').write_text(json.dumps(result,indent=2))
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case')
    args=parser.parse_args()
    print(json.dumps(validate(args.case),indent=2))
