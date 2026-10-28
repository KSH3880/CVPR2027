from types import SimpleNamespace
import torch
from ksh_jointpush.ksh_viewer_keys import install,ACTION


def task_fixture(viewer=True,evaluation=True):
    class Gym:
        def __init__(self):self.events=[];self.subscriptions=[]
        def subscribe_viewer_keyboard_event(self,*args):self.subscriptions.append(args)
        def query_viewer_action_events(self,viewer):
            events,self.events=self.events,[]
            return events
    class Task:
        def render(self):self.forwarded=self.gym.query_viewer_action_events(self.viewer)
        def _compute_reset(self):self.reset_buf.zero_();self._terminate_buf.fill_(1)
        def _compute_metrics_evaluation(self):self._success_buf.fill_(1);self._precision_buf.fill_(.1)
    install(Task)
    task=Task();task.viewer=object() if viewer else None;task._is_eval=evaluation;task.gym=Gym();task.num_envs=2
    task.reset_buf=torch.zeros(2,dtype=torch.long);task._terminate_buf=torch.zeros(2,4,dtype=torch.long)
    task._success_buf=torch.zeros(8,dtype=torch.long);task._precision_buf=torch.zeros(8)
    return task


def test_n_key_flows_through_scene_done_and_does_not_mark_success():
    task=task_fixture();gym=task.gym
    gym.events=[SimpleNamespace(action=ACTION,value=1),SimpleNamespace(action='follow',value=1)]
    task.render()
    assert task.gym is gym and len(gym.subscriptions)==1
    assert [x.action for x in task.forwarded]==['follow']
    task._compute_reset();task._compute_metrics_evaluation()
    assert task.reset_buf.tolist()==[1,0]
    assert task._terminate_buf[0].eq(0).all() and task._terminate_buf[1].eq(1).all()
    assert task._success_buf.tolist()==[0,0,0,0,1,1,1,1]
    assert torch.isinf(task._precision_buf[:4]).all()
    task.render();task._compute_reset();task._compute_metrics_evaluation()
    assert len(gym.subscriptions)==1 and task.reset_buf.eq(0).all()
    assert task._success_buf.eq(1).all()


def test_key_release_and_other_keys_do_not_skip():
    task=task_fixture();task.gym.events=[SimpleNamespace(action=ACTION,value=0),SimpleNamespace(action='QUIT',value=1)]
    task.render();task._compute_reset()
    assert task.reset_buf.eq(0).all()
    assert task.forwarded[0].action=='QUIT'


def test_headless_and_training_do_not_install_or_consume_keys():
    for viewer,evaluation in [(False,True),(True,False)]:
        task=task_fixture(viewer,evaluation);task.gym.events=[SimpleNamespace(action=ACTION,value=1)]
        task.render();task._compute_reset()
        assert not task.gym.subscriptions and task.reset_buf.eq(0).all()
        assert task.forwarded[0].action==ACTION
