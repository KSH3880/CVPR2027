"""Runtime bootstrap for the frozen KSH snapshot; no external code is patched."""
import os,sys,runpy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'ksh_jointpush'
sys.dont_write_bytecode=True
sys.path[:0]=[str(SOURCE/'tokenhsi'),str(SOURCE),str(ROOT)]
import isaacgym
import torch
import utils.output_paths as output_paths
_resolve=output_paths.resolve_output_path
_RUN_ROOT=ROOT/'runs/jointpush/ksh_jointpush_rsi'
output_paths.resolve_output_path=lambda experiment,mode,requested='',check=False,max_iterations=0,root=_RUN_ROOT:_resolve(experiment,mode,requested,check,max_iterations,root=_RUN_ROOT)
if os.environ.get('HEADLESS')=='0':
    if os.environ.get('CUDA_MPS_PIPE_DIRECTORY'):raise ValueError('Native viewer must bypass MPS')
    torch.cuda.set_device(int(os.environ['TOKENHSI_GUI_GPU_INDEX']))
    print('KSH_JOINTPUSH_NATIVE_GPU',os.environ['TOKENHSI_GUI_GPU_INDEX'],flush=True)
import yaml
prepared_env=yaml.safe_load(Path(sys.argv[sys.argv.index('--cfg_env')+1]).read_text())['env']
from ksh_jointpush.ksh_agent_rsi import install as install_agent_rsi
from ksh_jointpush.ksh_goal_cone import install as install_goal_cone
install_goal_cone(prepared_env)
if 'stage2JointPush' in prepared_env:
    from ksh_jointpush.ksh_soft_gate import install
    install(prepared_env)
if prepared_env.get('stage2JointPush',{}).get('coupled_rsi_selection')=='balanced_group_ternary_v1':
    from ksh_jointpush.ksh_balanced_rsi import install as install_balanced_rsi
    install_balanced_rsi(prepared_env)
else:
    install_agent_rsi(prepared_env)
from ksh_jointpush.ksh_push_only import install as install_push_only
install_push_only(prepared_env)
if os.environ.get('PUSH_RSI_MIX'):
    probability=float(os.environ['PUSH_RSI_MIX'])
    if '--test' not in sys.argv or not 0<=probability<=1:raise ValueError('PUSH_RSI_MIX is evaluation-only and must be in [0,1]')
    from env.tasks.multi_agent.humanoid_ma_stage2_jointpush_rsi import HumanoidMAStage2JointPushRsi
    from utils.stage2_jointpush_rsi import sample_push_initialization
    _sample_episode=HumanoidMAStage2JointPushRsi._sample_episode_graph
    def mixed_episode(self,ids):
        _sample_episode(self,ids)
        if not self._eval_push_rsi:raise ValueError('Mixed viewer requires --eval_push_rsi')
        chosen_probability=getattr(self,'_viewer_push_probability',probability)
        sampler=sample_push_initialization
        if prepared_env['stage2JointPush'].get('coupled_rsi_selection')=='independent_agents_v1':
            from ksh_jointpush.ksh_agent_rsi import sample_agent_initialization
            sampler=sample_agent_initialization
        if prepared_env['stage2JointPush'].get('coupled_rsi_selection')=='balanced_group_ternary_v1':
            from ksh_jointpush.ksh_balanced_rsi import sample_balanced_initialization
            sampler=sample_balanced_initialization
        self._interaction[ids]=sampler(self._groups[ids],self._templates_by_owner[ids],chosen_probability)
        print('KSH_VIEW_INIT',dict(push_probability=chosen_probability,push_people=int(self._interaction[ids].sum()),people=len(ids)*self.num_agents),flush=True)
    HumanoidMAStage2JointPushRsi._sample_episode_graph=mixed_episode
if '--test' not in sys.argv and 'stage2JointPush' in prepared_env:
    from ksh_jointpush.ksh_training_audit import install as install_audit
    install_audit()
if os.environ.get('HEADLESS')=='0' and os.environ.get('KSH_POSE_ONLY')!='1':
    from ksh_jointpush.ksh_viewer_keys import install as install_viewer_keys
    install_viewer_keys()
if os.environ.get('KSH_POSE_ONLY')=='1':
    if '--test' not in sys.argv:raise ValueError('Pose inspection is evaluation-only')
    from ksh_jointpush.ksh_pose_viewer import install as install_pose_viewer
    install_pose_viewer()
if os.environ.get('KSH_RSI_MODE_CHECK')=='1':
    from ksh_jointpush.tests.check_ksh_rsi_modes import install as install_rsi_check
    install_rsi_check()
entry=SOURCE/'tokenhsi/run.py'
if os.environ.get('KSH_JOINTPUSH_CHECK')=='1':
    entry=SOURCE/'tokenhsi/tests/check_stage2_jointpush_rsi.py'
    from env.tasks.multi_agent.humanoid_ma_stage2_jointpush import HumanoidMAStage2JointPush
    from ksh_jointpush.ksh_goal_cone import goal_in_cone
    _reset=HumanoidMAStage2JointPush._reset_envs
    def checked_reset(self,ids):
        result=_reset(self,ids)
        if len(ids):
            boxes=self._assigned_box_values(self._box_states,ids)
            active=self._templates_by_owner[ids]==5
            if prepared_env.get('stage2JointPush',{}).get('task_selection'):assert active.all()
            scene=prepared_env['jointPush']['scene']
            shared=self._members[ids]>1
            coupled_ok=goal_in_cone(boxes,self._tar_pos[ids],scene['goal_half_angle_degrees'])
            solo_ok=goal_in_cone(boxes,self._tar_pos[ids],scene.get('solo_goal_half_angle_degrees',scene['goal_half_angle_degrees']))
            assert (torch.where(shared,coupled_ok,solo_ok) | ~active).all()
            print('KSH_GOAL45_RESET_VERIFIED',dict(preset=self._task_graph_preset,push_people=int(active.sum())),flush=True)
        return result
    HumanoidMAStage2JointPush._reset_envs=checked_reset
runpy.run_path(str(entry),run_name='__main__')
