def active_eval_skills(config):
    return {name for name, weight in zip(config['skill'], config['skillInitProb']) if weight > 0}


def push_eval_skills(available, config, has_push, has_old):
    requested = active_eval_skills(config)
    needed = set()
    if has_old:
        if 'push' in requested:
            raise ValueError('PUSH RSI cannot initialize an old task; select TASK_GRAPH=push')
        needed.update(requested)
    if has_push:
        if requested - {'loco', 'push'}:
            raise ValueError('PUSH evaluation starts support loco and push')
        needed.update('push_loco' if name == 'loco' else name for name in requested)
    if needed - set(available):
        raise ValueError('Unknown evaluation motion skills: ' + str(sorted(needed - set(available))))
    return [name for name in available if name in needed]
