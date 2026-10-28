"""Original all-hands face gate with box-goal saturation preserved."""
from ksh_jointpush.ksh_soft_gate import SoftStage2JointPushRuntime


class HardStage2JointPushRuntime(SoftStage2JointPushRuntime):
    def step(self, phi, progress, z_error, feet_error=None, region_error=None):
        # The task computes the original all-members/both-hands 18cm gate.
        # Keep goal success independent of contact, as in the soft experiment.
        self.soft_gate.copy_(self.gate)
        result = super().step(phi, progress, z_error, feet_error, region_error)
        result.pop('soft_group_gate_raw')
        return result


def install(env):
    spec = env['stage2JointPush']['reward_gate']
    if (spec['mode'] != 'all_hands_face_tolerance_v1'
            or spec['success'] != 'box_goal_position'
            or spec['face_tolerance'] != .18
            or env['stage2JointPush']['gate_face_tolerance'] != .18):
        raise ValueError('Unknown local hard-gate contract')
    import env.tasks.multi_agent.stage2_jointpush_reward as reward
    reward.Stage2JointPushRuntime = HardStage2JointPushRuntime
    print('KSH_JOINTPUSH_HARD_GATE', spec, flush=True)
