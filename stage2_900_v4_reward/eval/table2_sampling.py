from copy import deepcopy

import torch


SETTINGS = dict(enable_sit=2, enable_climb=2, enable_ontop=2, coupled_2=2, coupled_3=3, coupled_4=4)


def prepare_bank(sizes, masses, env, preset, setting):
    people = env['numAgents']
    if preset != 'random_scenario' or people != SETTINGS[setting]:
        raise ValueError('Unexpected Table2 setting')
    n, device = len(sizes), sizes.device
    groups = torch.zeros(n, people, dtype=torch.long, device=device)
    parent = torch.full_like(groups, -1)
    templates = torch.full_like(groups, 5)
    domain = torch.ones(n, dtype=torch.long, device=device)
    if setting.startswith('enable_'):
        rows = torch.arange(n, device=device)
        leader = torch.randint(2, (n,), device=device)
        follower = 1-leader
        groups[:] = leader[:, None]
        templates[:] = 3
        templates[rows, follower] = dict(enable_sit=1, enable_climb=2, enable_ontop=4)[setting]
        parent[rows, follower] = leader
        domain.zero_()
        density = masses[rows, leader]/sizes[rows, leader].prod(-1)
        for axis, (lo, hi) in enumerate(((8,12), (8,12), (7,10))):
            sizes[rows, leader, axis] = torch.randint(lo, hi+1, (n,), device=device)/20
        masses[rows, leader] = density*sizes[rows, leader].prod(-1)
    return domain, groups, parent, templates


def install(setting):
    import json
    from pathlib import Path
    import stage2_protocol_sampling as sampling
    from env.tasks.multi_agent.humanoid_ma_stage2 import HumanoidMAStage2
    from utils import stage2_spec, stage2_eval_sizes
    from utils.coupled_push_spec import physical_box_count

    stage2_eval_sizes.prepare_intersection_bank = lambda sizes, masses, env, preset: prepare_bank(
        sizes, masses, env, preset, setting)
    original_validate = stage2_spec.validate_env

    def validate(env):
        if env['numAgents'] != 3:
            return original_validate(env)
        equivalent = deepcopy(env)
        equivalent.update(numAgents=4, numObjects=8)
        equivalent['stage2JointPush']['physical_boxes'] = physical_box_count(4, 8)
        if env['numObjects'] != 6:
            raise ValueError('Table2 three-person evaluation requires six tokens')
        original_validate(equivalent)

    stage2_spec.validate_env = validate
    original_boxes = HumanoidMAStage2._reset_boxes

    def unique_boxes(self, ids):
        slots = self._reset_ref_slots.get('loco')
        if slots is None:
            return original_boxes(self, ids)
        env, owner = slots
        keep = (self._members[env, owner] == 1) | (self._groups[env, owner] == owner)
        self._reset_ref_slots['loco'] = (env[keep], owner[keep])
        try:
            return original_boxes(self, ids)
        finally:
            self._reset_ref_slots['loco'] = slots

    HumanoidMAStage2._reset_boxes = unique_boxes
    original_capture = sampling.capture

    def capture(task):
        original_capture(task)
        n, m = task.num_envs, task.num_agents
        if m != SETTINGS[setting] or not (task._groups == task._groups[:, :1]).all():
            raise ValueError('Table2 scene contains more than one group')
        if setting.startswith('enable_'):
            if not ((task._before_predecessor >= 0).sum(-1) == 1).all():
                raise ValueError('Table2 scene is not an enabling pair')
        else:
            if (task._before_predecessor >= 0).any() or not (task._templates == 5).all():
                raise ValueError('Table2 scene is not pure coupled PUSH')
            if not (task._agent_box_assignment == task._agent_box_assignment[:, :1]).all():
                raise ValueError('Coupled agents do not share one physical box')
        path = Path(task.cfg['args'].output_path)/'new_conditions.json'
        conditions = json.loads(path.read_text())
        conditions.update(protocol='cvpr2027_table2_v1', setting=setting,
            group_sampler='table2_fixed_'+setting, solo_skill_probabilities=None,
            follower_probabilities=([float(setting == name) for name in ('enable_sit','enable_climb','enable_ontop')]
                                    if setting.startswith('enable_') else None))
        path.write_text(json.dumps(conditions, indent=2))

    sampling.capture = capture
