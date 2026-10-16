def reset_mixed(task, ids):
    if not len(ids):
        return
    joint = ids[task._joint_env_mask[ids]]
    independent = ids[~task._joint_env_mask[ids]]
    task._joint_reset.reset(joint, finalize=False)
    task._reset_ontop_context_envs(independent, finalize=False)
    task._joint_reset.commit(ids)
    task._refresh_sim_tensors()
    task._reset_relation_history(ids)
    task._compute_observations(ids)
    if len(independent):
        task._init_amp_obs(independent)
    if len(joint):
        task._compute_amp_observations(joint)
        task._hist_amp_obs_buf[joint] = task._curr_amp_obs_buf[joint, :, None]
