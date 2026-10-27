"""PUSH-only tasks/demo family while retaining original random 1..4-person groups."""
import torch


def push_templates(groups, probabilities=None):
    return torch.full_like(groups,5)


def fetch_push_demo(task,count):
    from utils.unified_training import append_family
    mapping=task.cfg['env']['ampExpertMapping']['push']
    skills=task.cfg['env']['skill']
    probability=mapping[skills.index('loco')]
    if abs(probability+mapping[skills.index('push')]-1)>1e-6:
        raise ValueError('PUSH expert must contain only loco/push')
    result=torch.empty(count,task.get_num_amp_obs(),device=task.device)
    loco=torch.rand(count,device=task.device)<probability
    for mask,name in ((loco,'push_loco'),(~loco,'push')):
        rows=mask.nonzero().flatten()
        if not len(rows):continue
        lib=task._motion_lib[name]
        mids=lib.sample_motions(len(rows));history=task.dt*(task._num_amp_obs_steps-1)
        times=lib.sample_time(mids,truncate_time=history)+history
        obs=task.build_amp_obs_demo(mids,times,lib).reshape(len(rows),task._num_amp_obs_steps,-1)
        label=torch.full((len(rows),task._num_amp_obs_steps),3,device=task.device,dtype=torch.long)
        result[rows]=append_family(obs,label,4).flatten(1)
    return result


def install(env):
    spec=env.get('stage2JointPush',{}).get('task_selection')
    if not spec:return
    if spec['mode']!='push_only_random_groups_v1':raise ValueError('Unknown task selection')
    from utils import stage2_jointpush
    from env.tasks.multi_agent.humanoid_ma_stage2_jointpush import HumanoidMAStage2JointPush
    stage2_jointpush.sample_templates=push_templates
    original=HumanoidMAStage2JointPush._sample_episode_graph
    def sample(self,ids):
        if self._task_graph_preset not in ('random_scenario','all_coupled','all_solo','push'):
            raise ValueError('PUSH-only configuration rejects other task presets')
        result=original(self,ids)
        assert (self._templates_by_owner[ids]==5).all()
        return result
    HumanoidMAStage2JointPush._sample_episode_graph=sample
    HumanoidMAStage2JointPush._fetch_conditioned_amp_demo=fetch_push_demo
    print('KSH_PUSH_ONLY_TASKS_AND_AMP',spec,flush=True)
