import time
from types import MethodType


def install(torch, gymapi, BaseTask, formal_eval, view_env):
    original_init = BaseTask.__init__
    original_eval = formal_eval.run_formal_eval
    session = {}

    def graphics_init(self, cfg, enable_camera_sensors=False):
        return original_init(self, cfg, enable_camera_sensors=True)

    BaseTask.__init__ = graphics_init

    def display_eval(player):
        task = player.env.task
        task.viewer = task.gym.create_viewer(task.sim, gymapi.CameraProperties())
        if task.viewer is None:
            raise RuntimeError('Could not create the formal VNC viewer')
        for key, action in ((gymapi.KEY_ESCAPE, 'QUIT'), (gymapi.KEY_SPACE, 'pause'),
                            (gymapi.KEY_F, 'follow')):
            task.gym.subscribe_viewer_keyboard_event(task.viewer, key, action)
        state = dict(paused=True, ready=False, env=view_env, camera_changed=True,
                     follow=False, finished=False)
        task._camera_follow_flag = False

        def camera(self):
            index = state['env']
            valid = self._stage2_task_slots[index]
            boxes = self._logical_box_values(self._box_states)[index, valid, :3]
            points = torch.cat((self._humanoid_root_states[index, :, :3], boxes,
                                self._tar_pos[index]), 0)
            low, high = points.amin(0), points.amax(0)
            center = (low + high) / 2
            center[2] = max(float(center[2]), .8)
            radius = max(float((high - low)[:2].norm()) * .8, 3.)
            self.gym.viewer_camera_look_at(self.viewer, None,
                gymapi.Vec3(float(center[0]) + radius * .65, float(center[1]) - radius,
                            2.8 + radius * .5), gymapi.Vec3(*center.tolist()))
            state['camera_changed'] = False

        def render(self, sync_frame_time=False):
            if not state['ready']:
                return
            while True:
                started = time.monotonic()
                if self.gym.query_viewer_has_closed(self.viewer):
                    raise SystemExit(0)
                for event in self.gym.query_viewer_action_events(self.viewer):
                    if event.value <= 0:
                        continue
                    if event.action == 'QUIT':
                        raise SystemExit(0)
                    if event.action == 'pause' and not state['finished']:
                        state['paused'] = not state['paused']
                        print('[formal VNC] ' + ('paused' if state['paused'] else 'running'), flush=True)
                    if event.action == 'follow':
                        state['follow'] = not state['follow']
                if state['follow'] or state['camera_changed']:
                    camera(self)
                BaseTask.render(self, sync_frame_time)
                time.sleep(max(0., 1 / 60 - (time.monotonic() - started)))
                if not state['paused']:
                    return

        task.render = MethodType(render, task)
        reset = player.env_reset

        def reset_and_show(*values, **kwargs):
            observation = reset(*values, **kwargs)
            state['ready'] = True
            print('[formal VNC] scene={} ready; click viewer and press Space to start'.format(state['env']), flush=True)
            task.render()
            return observation

        player.env_reset = reset_and_show
        session.update(task=task, state=state)
        try:
            return original_eval(player)
        finally:
            player.env_reset = reset

    formal_eval.run_formal_eval = display_eval

    def hold():
        print('[formal VNC] 900-step scene finished; Esc: exit', flush=True)
        session['state'].update(paused=True, finished=True)
        session['task'].render()

    return hold
