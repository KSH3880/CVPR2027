from utils.push_distance_eval import summarize_distance_results


def test_distance_bins_boundaries_and_success_denominators():
    rows = [dict(initial_goal_distance_m=d, initial_agent_box_distance_m=1.,
                 final_goal_distance_m=.1 if i % 2 == 0 else 2.,
                 ever_success=i % 2 == 0, final_success=False,
                 first_success_seconds=3. if i % 2 == 0 else None)
            for i, d in enumerate((3., 3.5, 4., 4.5, 5., 2.9))]
    s = summarize_distance_results({'repeat_0': dict(num_scene_trials=3,
                                   push_distance_audit=dict(agents=rows))})
    assert s['overall']['agent_trials'] == 6
    assert s['overall']['ever_success_rate'] == .5
    assert s['overall']['final_success_rate'] == 0.
    assert [v['agent_trials'] for v in s['by_initial_box_goal_xy_distance'].values()] == [1, 1, 1, 2]
    assert s['outside_3to5m']['agent_trials'] == 1
    assert s['overall']['mean_first_success_seconds'] == 3.


def test_empty_distance_summary():
    s = summarize_distance_results({})
    assert s['overall']['agent_trials'] == 0
    assert s['overall']['ever_success_rate'] is None


def test_first_episode_masks_and_incremental_save(monkeypatch, tmp_path):
    import json
    from types import SimpleNamespace as NS
    import torch
    from learning.multi_agent import edge_context_eval as ev

    monkeypatch.setenv('TOKENHSI_PUSH_DISTANCE_EVAL', '1')
    monkeypatch.delenv('TOKENHSI_EVAL_TRAIN_RSI', raising=False)
    monkeypatch.delenv('TOKENHSI_PUSH_FACE_AUDIT', raising=False)
    monkeypatch.delenv('TOKENHSI_PUSH_EVAL_AUDIT', raising=False)
    graph = NS(ids=['a', 'b'], required_goal=torch.ones(2, 2, dtype=torch.bool),
               edge_valid=torch.ones(2, 2, dtype=torch.bool), edge_relation=torch.ones(2, 2))
    monkeypatch.setattr(ev, 'owner_sum', lambda x, g: x)
    monkeypatch.setattr(ev, 'scene_success', lambda x, g: x.all(-1))
    runtime = NS(graph=graph, done=torch.zeros(2, 2, dtype=torch.bool),
                 own_success=torch.zeros(2, 2, dtype=torch.bool))
    boxes = torch.zeros(2, 2, 13)
    goals = torch.zeros(2, 2, 3); goals[..., 0] = 3.5
    task = NS(num_envs=2, num_agents=2, relation_runtime=runtime, _open_push=True,
              _rsi_interaction=torch.zeros(2, 2, dtype=torch.bool), _tar_pos=goals,
              _humanoid_root_states=boxes.clone(), _owned_push_states=lambda: boxes,
              dt=.1, max_episode_length=3,
              cfg={'args': NS(eval_repeats=1, output_path=str(tmp_path))})

    class Player:
        device = 'cpu'
        is_determenistic = True
        env = NS(task=task)
        step = 0

        def env_reset(self, *args):
            return {'obs': torch.zeros(4, 1)}

        def get_batch_size(self, *args):
            pass

        def get_action(self, *args):
            return None

        def env_step(self, *args):
            self.step += 1
            runtime.done[:] = False
            runtime.done[0] = self.step == 1
            runtime.done[1] = self.step == 2
            runtime.own_success.copy_(runtime.done)
            boxes[0, :, 0] = 3.4 if self.step == 1 else -10.
            boxes[1, :, 0] = 3.4 if self.step == 2 else 0.
            done = torch.tensor([[self.step == 1]*2, [self.step == 3]*2]).flatten()
            return None, None, done, {'terminate': torch.zeros(4, dtype=torch.bool)}

    result = ev.run_edge_context_eval(Player())
    rows = result['repeat_0']['push_distance_audit']['agents']
    assert all(r['ever_success'] for r in rows)
    assert [r['final_success'] for r in rows] == [True, True, False, False]
    assert abs(rows[0]['final_goal_distance_m']-.1) < 1e-5
    assert abs(rows[0]['first_success_seconds']-.1) < 1e-5
    assert abs(rows[2]['first_success_seconds']-.2) < 1e-5
    saved = json.loads((tmp_path/'metrics/distance_summary.json').read_text())
    assert saved['overall']['ever_success_rate'] == 1.
    assert saved['overall']['final_success_rate'] == .5
