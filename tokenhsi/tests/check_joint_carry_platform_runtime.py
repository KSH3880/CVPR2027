import json
import runpy
import sys
from pathlib import Path

import isaacgym
import torch
from isaacgym import gymtorch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
context=runpy.run_path(str(ROOT/'tests/check_joint_carry_goal360_runtime.py'))
t=context['task']
args=context['context']['args']
p=t._joint_loco_platform
ids=t._joint_env_mask.nonzero().flatten()
solo=(~t._joint_env_mask).nonzero().flatten()
weights=t._template_rsi_weights.clone()
preset=t._task_graph_preset
report=dict(environments=t.num_envs,joint=len(ids),independent=len(solo),resets=[],physics=[])

try:
    for name in ('joint_carry_at','joint_carry_ontop'):
        t._task_graph_preset=name
        for skill in ('loco','pickUp','carryWith','putDown'):
            if skill=='putDown' and name=='joint_carry_ontop':
                continue
            t._template_rsi_weights.zero_()
            t._template_rsi_weights[:,t._skill.index(skill)]=1.
            t._joint_loco_platform=None
            torch.manual_seed(8485)
            t._reset_envs(ids)
            old=[x[ids].clone() for x in (t._humanoid_root_states,t._dof_pos,t._dof_vel,
                 t._box_states,t._tar_pos,t._amp_obs_buf)]
            untouched=t._root_states.view(t.num_envs,-1,13)[solo].clone()
            t._joint_loco_platform=p
            torch.manual_seed(8485)
            t._reset_envs(ids)
            for a,b in zip(old[:3],(t._humanoid_root_states,t._dof_pos,t._dof_vel)):
                torch.testing.assert_close(a,b[ids],rtol=0,atol=0)
            torch.testing.assert_close(old[4],t._tar_pos[ids],rtol=0,atol=0)
            torch.testing.assert_close(old[5],t._amp_obs_buf[ids],rtol=0,atol=0)
            torch.testing.assert_close(untouched,t._root_states.view(t.num_envs,-1,13)[solo],rtol=0,atol=0)
            delta=t._box_states[ids]-old[3]
            expected=torch.zeros_like(delta)
            expected[:,0,2]=p.height[ids]+p.active[ids]*.05
            torch.testing.assert_close(delta,expected,rtol=0,atol=2e-6)
            assert not p.active[solo].any() and (p.states[solo,2]==20.).all()
            active=ids[p.active[ids]]
            item=dict(task=name,skill=skill,active=len(active),count=len(ids))
            if skill=='loco':
                assert abs(len(active)/len(ids)-.5)<.04
                h=p.height[active]
                maximum=1.2-t._box_size[active,0,2]
                assert (h>=.02).all() and (h<=maximum).all()
                at_max=torch.isclose(h,maximum)
                assert abs(float(at_max.float().mean())-.2)<.05
                torch.testing.assert_close(p.states[active,:2],t._box_states[active,0,:2])
                torch.testing.assert_close(p.states[active,3:7],t._box_states[active,0,3:7])
                torch.testing.assert_close(t._box_states[active,0,2]-t._box_size[active,0,2]/2-
                                          (p.states[active,2]+.01),h.new_full(h.shape,.05),atol=2e-6,rtol=0)
                item.update(height_min=float(h.min()),height_max=float(h.max()),max_fraction=float(at_max.float().mean()))
            else:
                assert not len(active) and (p.states[ids,2]==20.).all()
            report['resets'].append(item)

        t._template_rsi_weights.zero_()
        t._template_rsi_weights[:,t._skill.index('loco')]=1.
        t._reset_envs(ids)
        remaining=ids[1::2]
        old=t._root_states.view(t.num_envs,-1,13)[remaining].clone()
        old_active=p.active[remaining].clone()
        t._reset_envs(ids[::2])
        torch.testing.assert_close(old,t._root_states.view(t.num_envs,-1,13)[remaining],atol=0,rtol=0)
        assert torch.equal(old_active,p.active[remaining])
        t._humanoid_root_states[ids,:,0]+=10.
        t._joint_reset.commit(ids)
        active=ids[p.active[ids]]
        initial=t._box_states[active,0,:3].clone()
        pad_initial=p.states[active,:3].clone()
        for _ in range(120):
            t.gym.simulate(t.sim);t.gym.fetch_results(t.sim,True)
        t._refresh_sim_tensors()
        gap=t._box_states[active,0,2]-t._box_size[active,0,2]/2-(p.states[active,2]+.01)
        drift=(t._box_states[active,0,:2]-initial[:,:2]).norm(dim=-1)
        pad_drift=(p.states[active,:3]-pad_initial).norm(dim=-1)
        assert gap.abs().max()<.003 and drift.max()<.02 and pad_drift.max()<.003
        item=dict(task=name,supported=len(active),gap_max=float(gap.abs().max()),
                  payload_xy_drift_max=float(drift.max()),platform_drift_max=float(pad_drift.max()))
        high=active[p.height[active]>.15]
        z=t._box_states[high,0,2].clone()
        p.states[high,2]=20.
        p.states[high,7:]=0.
        actors=p.actor_ids[high].contiguous()
        t.gym.set_actor_root_state_tensor_indexed(t.sim,gymtorch.unwrap_tensor(t._root_states),
                                                gymtorch.unwrap_tensor(actors),len(actors))
        for _ in range(60):
            t.gym.simulate(t.sim);t.gym.fetch_results(t.sim,True)
        t._refresh_sim_tensors()
        fall=z-t._box_states[high,0,2]
        assert (fall>.1).all()
        item['removed_platform_fall_min']=float(fall.min())
        report['physics'].append(item)
        print('PLATFORM_PHYSICS',json.dumps(item),flush=True)
finally:
    t._joint_loco_platform=p
    t._template_rsi_weights.copy_(weights)
    t._task_graph_preset=preset
report['partial_reset_preserved']=True
report['only_loco_payload_changes']=True
out=Path(args.output_path)
(out/'platform_runtime_check.json').write_text(json.dumps(report,indent=2)+'\n')
print('PLATFORM_RUNTIME',json.dumps(report),flush=True)
