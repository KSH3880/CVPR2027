import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from tensorboard_history import write_history
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.summary.writer.record_writer import RecordWriter
from tensorboard.backend.event_processing.event_file_loader import RawEventFileLoader


def test_remove_abandoned_steps_on_all_axes(tmp_path):
    source = tmp_path / 'events.source'
    with source.open('wb') as stream:
        writer = RecordWriter(stream)
        for epoch in (10, 11):
            for tag, step, val, delta in [('info/epochs', epoch*100, epoch, 0),
                                           ('reward_push/frame', epoch*100, 1, .1),
                                           ('reward_push/iter', epoch, 1, .2),
                                           ('reward_push/time', epoch*5, 1, .3)]:
                e = Event(step=step, wall_time=epoch+delta)
                e.summary.value.add(tag=tag, simple_value=val)
                writer.write(e.SerializeToString())
        writer.flush()
    dest = tmp_path / 'events.history'
    assert write_history([source], dest, 10, 20) == 70
    got = [(v.tag, e.step) for raw in RawEventFileLoader(str(dest)).Load()
           for e in [Event.FromString(raw)] for v in e.summary.value]
    assert got == [('info/epochs', 1000), ('reward_push/frame', 1000),
                   ('reward_push/iter', 10), ('reward_push/time', 70)]
