from utils.before_spec import check_checkpoint, checkpoint_contract
from utils.experiments import BEFORE_PUSH_RELFUSION, BEFORE_TOTAL_FT21K, native_config


def initialize(agent, checkpoint, cfg):
    target = native_config(BEFORE_TOTAL_FT21K)['env']
    if cfg['experiment']['name'] != BEFORE_TOTAL_FT21K or cfg['env']['episodeLength'] != 900:
        raise ValueError('Stage2 weight initialization requires the total_ft21k experiment')
    check_checkpoint({'before_contract': checkpoint_contract(target)}, cfg['env'])
    source = native_config(BEFORE_PUSH_RELFUSION)['env']
    check_checkpoint(checkpoint, source)
    if checkpoint['epoch'] != 21000:
        raise ValueError('total_ft21k requires fusion v1 epoch 21000')
    if checkpoint['relation_experiment_config']['env']['relationGraph'] != source['relationGraph']:
        raise ValueError('Fine-tuning source graph differs from fusion v1')
    agent.model.load_state_dict(checkpoint['model'], strict=True)
    agent.running_mean_std.load_state_dict(checkpoint['running_mean_std'], strict=True)
    agent._amp_input_mean_std.load_state_dict(checkpoint['amp_input_mean_std'], strict=True)
    if agent.normalize_value:
        agent.value_mean_std.load_state_dict(checkpoint['reward_mean_std'], strict=True)
    return dict(initialization='stage2_weights', source_epoch=21000,
                copied=list(checkpoint['model']), optimizer_restored=False,
                optimizer_state_entries=len(agent.optimizer.state),
                initial_epoch=agent.epoch_num, initial_frame=agent.frame)
