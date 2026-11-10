import argparse
import json
import os
from pathlib import Path
from .client import OneShotScheduler, chat_transport, loads
from .core import build_messages


def main():
    parser = argparse.ArgumentParser(description='CPU-only one-shot scheduler (does not launch simulation)')
    parser.add_argument('request', type=Path)
    parser.add_argument('--response-file', type=Path, help='Validate an already recorded LLM response offline')
    parser.add_argument('--endpoint', help='Explicit chat-completions-compatible URL')
    parser.add_argument('--model')
    parser.add_argument('--output', type=Path, help='New directory under output_etc/check/')
    parser.add_argument('--messages-only', action='store_true')
    args = parser.parse_args()
    request = loads(args.request.read_text())
    if args.messages_only:
        print(json.dumps(build_messages(request), ensure_ascii=False, indent=2))
        return
    if args.output is None:
        parser.error('--output is required')
    if bool(args.response_file) == bool(args.endpoint):
        parser.error('Choose exactly one of --response-file or --endpoint')
    if args.response_file:
        raw = args.response_file.read_text()
        transport = lambda messages: (raw, {'mode': 'offline_replay', 'network_calls': 0})
    else:
        if not args.model:
            parser.error('--model is required with --endpoint')
        transport = chat_transport(args.endpoint, args.model, os.environ.get('SCT_LLM_API_KEY'))
    answer = OneShotScheduler(transport).schedule(request, args.output)
    print(json.dumps(answer, ensure_ascii=False, indent=2))
    if answer['status'] != 'ok':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
