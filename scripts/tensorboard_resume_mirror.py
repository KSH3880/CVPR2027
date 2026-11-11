"""Mirror live resume events, offsetting /time steps without editing source logs."""
import argparse
import json
import signal
import time
from pathlib import Path

from tensorboard.backend.event_processing.event_file_loader import RawEventFileLoader
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.summary.writer.record_writer import RecordWriter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    args = parser.parse_args()
    manifest = Path(args.manifest)
    config = json.loads(manifest.read_text())
    destination = manifest.parent / 'events.out.tfevents.9999999999.resume_time_corrected'
    offset = config['resume_time_offset_seconds']
    loaders = {}
    running = True

    def stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    # Rebuild the derived stream on restart; original event files remain intact.
    with destination.open('wb') as stream:
        writer = RecordWriter(stream)
        while running:
            for source in sorted(Path(config['active_summaries']).glob('events.out.tfevents.*')):
                if source not in loaders:
                    loaders[source] = RawEventFileLoader(str(source))
                for raw in loaders[source].Load():
                    event = Event.FromString(raw)
                    if event.HasField('summary'):
                        # Split mixed summaries so only time-based tags change step.
                        for value in event.summary.value:
                            copy = Event()
                            copy.CopyFrom(event)
                            del copy.summary.value[:]
                            copy.summary.value.add().CopyFrom(value)
                            if value.tag.endswith('/time'):
                                copy.step += offset
                            writer.write(copy.SerializeToString())
                    else:
                        writer.write(raw)
            writer.flush()
            time.sleep(2)


if __name__ == '__main__':
    main()
