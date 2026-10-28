"""Viewer-only N-key episode skip through the normal scene-done/reset path."""
from isaacgym import gymapi

ACTION = 'ksh_next_episode'


class EventFilter:
    def __init__(self, gym, task):
        self.gym, self.task = gym, task

    def __getattr__(self, name):
        return getattr(self.gym, name)

    def query_viewer_action_events(self, viewer):
        result=[]
        for event in self.gym.query_viewer_action_events(viewer):
            if event.action == ACTION:
                if event.value > 0:
                    self.task._viewer_next_requested = True
            else:
                result.append(event)
        return result


def install(task_class=None):
    if task_class is None:
        from env.tasks.multi_agent.humanoid_ma_stage2_jointpush_rsi import HumanoidMAStage2JointPushRsi
        task_class=HumanoidMAStage2JointPushRsi
    if task_class.__dict__.get('_ksh_next_episode_installed',False):
        return
    original_render=task_class.render
    original_reset=task_class._compute_reset
    original_metrics=task_class._compute_metrics_evaluation

    def render(self,*args,**kwargs):
        if not self.viewer or not self._is_eval:
            return original_render(self,*args,**kwargs)
        if not getattr(self,'_viewer_next_subscribed',False):
            self.gym.subscribe_viewer_keyboard_event(self.viewer,gymapi.KEY_N,ACTION)
            self._viewer_next_subscribed=True
            print('KSH_VIEWER_KEYS N=next episode, Esc=quit',flush=True)
        gym=self.gym
        self.gym=EventFilter(gym,self)
        try:
            return original_render(self,*args,**kwargs)
        finally:
            self.gym=gym

    def reset(self,*args,**kwargs):
        result=original_reset(self,*args,**kwargs)
        if self.viewer and self._is_eval and getattr(self,'_viewer_next_requested',False):
            # reset_buf is per scene; VecTask expands done to all four agents.
            self.reset_buf[0]=1
            self._terminate_buf.reshape(self.num_envs,-1)[0]=0
            self._viewer_next_requested=False
            self._viewer_manual_skip=True
            print('KSH_VIEWER_NEXT_EPISODE scene=0 manual_skip=1',flush=True)
        return result

    def metrics(self,*args,**kwargs):
        result=original_metrics(self,*args,**kwargs)
        if getattr(self,'_viewer_manual_skip',False):
            self._success_buf.reshape(self.num_envs,-1)[0]=0
            self._precision_buf.reshape(self.num_envs,-1)[0]=float('inf')
            self._viewer_manual_skip=False
        return result

    task_class.render=render
    task_class._compute_reset=reset
    task_class._compute_metrics_evaluation=metrics
    task_class._ksh_next_episode_installed=True
