"""Copy only the checkpointed branch of a training event stream."""
from tensorboard.backend.event_processing.event_file_loader import RawEventFileLoader
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.summary.writer.record_writer import RecordWriter


def write_history(sources, destination, through_iter, offset):
    epochs = []
    for source in sources:
        for raw in RawEventFileLoader(str(source)).Load():
            e = Event.FromString(raw)
            for v in e.summary.value:
                if v.tag == 'info/epochs':
                    epochs.append((v.simple_value, e.step, e.wall_time))
    included = [e for e in epochs if e[0] <= through_iter]
    if not included:
        raise ValueError('No epoch records at checkpoint boundary')
    max_frame = max(e[1] for e in included)
    cutoff = min((e[2] for e in epochs if e[0] > through_iter), default=float('inf'))
    end = offset
    with destination.open('wb') as stream:
        writer = RecordWriter(stream)
        for source in sources:
            for raw in RawEventFileLoader(str(source)).Load():
                e = Event.FromString(raw)
                if not e.HasField('summary'):
                    if e.HasField('file_version'):
                        writer.write(raw)
                    continue
                for v in e.summary.value:
                    if v.tag.endswith('/time'):
                        keep = e.wall_time < cutoff
                    elif v.tag.endswith('/iter'):
                        keep = e.step <= through_iter
                    else:
                        keep = e.step <= max_frame
                    if not keep:
                        continue
                    copy = Event()
                    copy.CopyFrom(e)
                    del copy.summary.value[:]
                    copy.summary.value.add().CopyFrom(v)
                    if v.tag.endswith('/time'):
                        copy.step += offset
                        end = max(end, copy.step)
                    writer.write(copy.SerializeToString())
        writer.flush()
    return end
