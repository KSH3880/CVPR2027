"""Mirror PUSH-only scalars and add CONTACT/AT rewards without touching source events.

TensorBoard must read the mirror, not two concurrently written event files in
one directory: its directory loader can stop tailing the earlier writer.
"""
import argparse
import json
from pathlib import Path
import time
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter

COMPONENTS = dict(state='edge_state', progress='edge_progress', success='edge_success')


def convert(logdir, writer, last_step, copied):
    events = EventAccumulator(str(logdir), size_guidance={'scalars': 0})
    events.Reload()
    values = {tag: {row.step: row for row in events.Scalars(tag)}
              for tag in events.Tags()['scalars']}
    sources = ['relation/agent/task_total', 'relation/edge/at/total']
    for component, name in COMPONENTS.items():
        sources += ['reward_terms/'+name, 'relation/edge/at/'+component+'_component']
    pending = {}
    for tag, rows in values.items():
        for step, row in rows.items():
            if step > copied.get(tag, -1):
                pending.setdefault((step, row.wall_time), []).append(
                    Summary.Value(tag=tag, simple_value=row.value))
                copied[tag] = max(copied.get(tag, -1), step)
    count = 0
    if set(sources).issubset(values):
        steps = sorted(set.intersection(*(set(values[tag]) for tag in sources)))
        for step in steps:
            if step <= last_step:
                continue
            def value(tag):
                return values[tag][step].value
            total = value('relation/agent/task_total')
            summed = sum(value('reward_terms/'+name) for name in COMPONENTS.values())
            if abs(total-summed) > 1e-4:
                raise ValueError('Component aggregation differs at step '+str(step))
            output = []
            for component, name in list(COMPONENTS.items()) + [('total', None)]:
                at = value('relation/edge/at/'+('total' if component == 'total' else component+'_component'))
                contact = (total if name is None else value('reward_terms/'+name)) - at
                if contact < -1e-4:
                    raise ValueError('Invalid CONTACT subtraction at step '+str(step))
                for edge, result in [('contact', contact), ('at', at)]:
                    output.append(Summary.Value(tag='reward_terms/'+edge+'_'+component, simple_value=result))
            pending.setdefault((step, values[sources[0]][step].wall_time), []).extend(output)
            last_step = step
            count += 1
    for (step, wall_time), scalars in sorted(pending.items()):
        writer.add_event(Event(wall_time=wall_time, step=step, summary=Summary(value=scalars)))
    writer.flush()
    return last_step, count, copied


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--logdir', type=Path, required=True)
    parser.add_argument('--output-logdir', type=Path, required=True)
    parser.add_argument('--env-config', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--watch-pid', type=int)
    args = parser.parse_args()
    import yaml
    env = yaml.safe_load(args.env_config.read_text())['env']
    if env['stage2JointPush']['task_selection']['mode'] != 'push_only_random_groups_v1':
        raise ValueError('Reward splitting requires PUSH-only per-agent CONTACT+AT')
    if args.logdir.resolve() == args.output_logdir.resolve():
        raise ValueError('Source and mirror must use separate directories')
    state = json.loads(args.state.read_text()) if args.state.exists() else {}
    last, copied = state.get('last_step', -1), state.get('copied', {})
    writer = EventFileWriter(str(args.output_logdir), filename_suffix='.contact_at_split')
    try:
        while True:
            last, count, copied = convert(args.logdir, writer, last, copied)
            args.state.write_text(json.dumps(dict(last_step=last, newly_written=count, copied=copied))+'\n')
            print('CONTACT_AT_SPLIT', last, count, flush=True)
            if not args.watch_pid or not Path('/proc', str(args.watch_pid)).exists():
                break
            time.sleep(30)
    finally:
        writer.close()


if __name__ == '__main__':
    main()
