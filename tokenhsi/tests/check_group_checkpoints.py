import argparse
import json
from pathlib import Path

import torch
from learning.multi_agent.scene_normalizer import SceneRunningMeanStd
from learning.multi_agent.grasp_view import initial_grasp_sides, make_grasp_views
from learning.multi_agent.stage2_transfer import transfer_stage1_weights
from utils.joint_carry_amp import check_training_checkpoint
from utils.relation_task_spec import check_checkpoint_metadata, checkpoint_metadata
from sa_helpers import Wrapper, configs
from test_group_carry import network, scene


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('checkpoints', nargs='+')
    args = parser.parse_args()
    source = torch.load(args.source, map_location='cpu', weights_only=False)
    reports = []
    for filename in args.checkpoints:
        saved = torch.load(filename, map_location='cpu', weights_only=False)
        grasp = saved.get('joint_carry_grasp_view') is not None
        name = 'stage2_grasp_at20_4agent' if grasp else 'stage2_at20_4agent'
        env = configs(name)[0]
        check_training_checkpoint(saved, env)
        check_checkpoint_metadata(saved, checkpoint_metadata(env['relationReward']))
        model = Wrapper(network(grasp)).eval()
        transfer_stage1_weights(model, source)
        model.load_state_dict(saved['model'], strict=True)
        for key, value in saved['model'].items():
            if not torch.isfinite(value).all():
                raise RuntimeError('Nonfinite model tensor: '+key)
        groups = {'actor_encoder': 0, 'critic_encoder': 0, 'action_head': 0,
                  'value_head': 0, 'coordination': 0, 'critic_coordination': 0}
        for key, value in saved['model'].items():
            block = key.split('.')[1]
            if block not in groups:
                continue
            if block == 'actor_encoder':
                torch.testing.assert_close(value, source['model'][key], atol=0, rtol=0)
                groups[block] += 1
            elif block in ('coordination', 'critic_coordination'):
                if key.endswith('relation_bias'):
                    if value.abs().sum() == 0:
                        raise RuntimeError('SA relation bias did not update: '+key)
                    groups[block] += 1
            elif not torch.equal(value, source['model'][key]):
                groups[block] += 1
        if any(count == 0 for count in groups.values()):
            raise RuntimeError('Expected frozen/updated block was absent: '+str(groups))
        rms = SceneRunningMeanStd([223, 30, 1], [4, 8, 4], [223, 30, 0],
            kinematic_size=7, extra_passthrough_size=20, num_views=4 if grasp else 1).eval()
        rms.load_state_dict(saved['running_mean_std'], strict=True)
        for key, value in saved['running_mean_std'].items():
            torch.testing.assert_close(value, source['running_mean_std'][key], atol=0, rtol=0)
        obs, _ = scene(n=8)
        if grasp:
            obs = make_grasp_views(obs, initial_grasp_sides(obs, 4, 8), .15)
        with torch.no_grad():
            obs = rms(obs)
            actions, sigma = model.a2c_network.eval_actor(obs)
            values = model.a2c_network.eval_critic(obs)
        if actions.shape != (32, 32) or not all(torch.isfinite(x).all() for x in (actions, sigma, values)):
            raise RuntimeError('Reloaded actor/critic forward failed')
        reports.append(dict(path=filename, epoch=saved['epoch'], groups=groups,
            rms_unchanged=True, all_model_finite=True, action_shape=list(actions.shape),
            obs_width=obs.shape[-1]))
    Path(args.output).write_text(json.dumps(reports, indent=2))
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
