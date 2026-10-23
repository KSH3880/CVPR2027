"""Evaluation-only physical box friction override, preserving checkpoint contracts."""
from isaacgym import gymapi
import json,math,os,runpy
from pathlib import Path
from env.tasks.multi_agent.humanoid_ma_open_push import HumanoidMAOpenPush

friction=float(os.environ.get('BOX_FRICTION','0.6'))
if not math.isfinite(friction) or friction not in (.2,.4,.6):
    raise ValueError('BOX_FRICTION must be 0.2, 0.4 or 0.6')
original=HumanoidMAOpenPush._build_box

def build(self,env_id,object_id,env_ptr):
    args=self.cfg['args']
    if not (args.test and args.eval):raise ValueError('Friction override is evaluation-only')
    original(self,env_id,object_id,env_ptr)
    if object_id<2:return
    handle=self._box_handles[-1]
    props=self.gym.get_actor_rigid_shape_properties(env_ptr,handle)
    for prop in props:prop.friction=friction
    self.gym.set_actor_rigid_shape_properties(env_ptr,handle,props)
    actual=self.gym.get_actor_rigid_shape_properties(env_ptr,handle)
    assert all(abs(prop.friction-friction)<1e-6 for prop in actual)
    if env_id==0:
        out=Path(args.output_path);out.mkdir(parents=True,exist_ok=True)
        metadata=dict(checkpoint=args.checkpoint,seed=args.seed,evaluation_only=True,
            checkpoint_config_friction=self.cfg['env']['pushBox']['friction'],box_friction=friction,
            actual_shape_friction=[p.friction for p in actual],plane=self.cfg['env']['plane'])
        (out/'friction_override.json').write_text(json.dumps(metadata,indent=2)+'\n')
        print('EVAL_BOX_FRICTION',json.dumps(metadata),flush=True)

HumanoidMAOpenPush._build_box=build
runpy.run_path('tokenhsi/run.py',run_name='__main__')
