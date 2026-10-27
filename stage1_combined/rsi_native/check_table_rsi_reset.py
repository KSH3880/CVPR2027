"""2048-env corrected-history and partial-reset integration check, no training."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tokenhsi'),str(ROOT)]
import isaacgym
import torch,numpy as np
from learning.multi_agent.ma_players import MAPlayerContinuous


def check(player):
    t=player.env.task;player.env_reset();pool=t._table_carry_rsi
    active=pool.active.clone();assert active.any() and (~active).any()
    assert torch.equal(t._hist_amp_obs_buf[active,0],pool.history[active])
    assert torch.isfinite(t._amp_obs_buf).all()
    ids=torch.arange(0,t.num_envs,2,device=t.device);other=torch.arange(1,t.num_envs,2,device=t.device)
    before=[v[other].clone() for v in (t._humanoid_root_states,t._box_states,t._dof_pos,t._hist_amp_obs_buf)]
    t.reset(ids)
    for original,v in zip(before,(t._humanoid_root_states,t._box_states,t._dof_pos,t._hist_amp_obs_buf)):
        assert torch.equal(original,v[other]), 'Partial reset changed other environments'
    assert torch.equal(t._hist_amp_obs_buf[pool.active,0],pool.history[pool.active])
    out=Path(t.cfg['args'].output_path);out.mkdir(parents=True,exist_ok=True)
    rows=pool.active.nonzero().flatten()[:12]
    np.savez_compressed(out/'poses.npz',root=t._humanoid_root_states[rows,0].cpu().numpy(),body=t._kinematic_humanoid_rigid_body_states[rows,0].cpu().numpy(),objects=t._box_states[rows,0].cpu().numpy(),origins=t._env_origins[rows].cpu().numpy(),model_ids=t._object_model_ids[rows].cpu().numpy(),sample_ids=pool.sample[rows].cpu().numpy())
    report=dict(num_envs=t.num_envs,active_rsi=int(pool.active.sum()),requested=pool.requested,applied=pool.applied,explicit_loco_fallbacks=pool.missing,history_exact=True,amp_finite=True,partial_reset_exact=True,optimizer_steps=0,keys=pool.keys)
    (out/'validation.json').write_text(json.dumps(report,indent=2));print('TABLE_RSI_RESET',json.dumps(report),flush=True)

MAPlayerContinuous.run_eval=check
import run as app
try:app.main()
finally:
    if torch.cuda.is_initialized():torch.cuda.synchronize()
