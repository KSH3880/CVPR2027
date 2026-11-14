import argparse
import json
from pathlib import Path
from .client import OneShotScheduler, chat_transport, loads
from .core import build_messages
from .config import DEFAULT_BASE_URL, DEFAULT_REASONING_EFFORT, REASONING_EFFORTS, chat_endpoint, read_settings


def main():
    parser = argparse.ArgumentParser(description='CPU-only one-shot scheduler (does not launch simulation)')
    parser.add_argument('request', type=Path)
    parser.add_argument('--response-file', type=Path, help='Validate an already recorded LLM response offline')
    urls = parser.add_mutually_exclusive_group()
    urls.add_argument('--endpoint', help='Full chat-completions-compatible URL (override)')
    urls.add_argument('--base-url', help='Base URL; defaults to OpenAI or configured provider')
    parser.add_argument('--env-file', type=Path, help='Literal settings file; default: project .env')
    parser.add_argument('--json-mode', action='store_true', help='Send optional response_format=json_object')
    parser.add_argument('--model')
    parser.add_argument('--reasoning-effort', choices=REASONING_EFFORTS,
                        help='Overrides SCT_LLM_REASONING_EFFORT; missing/blank defaults to medium')
    parser.add_argument('--output', type=Path, help='New directory under output_etc/check/')
    parser.add_argument('--messages-only', action='store_true')
    parser.add_argument('--fewshot', action='store_true', help='Attach the ten recorded Stage1 references')
    args = parser.parse_args()
    request = loads(args.request.read_text())
    if args.fewshot:
        from .fewshot import attach_references
        request = attach_references(request)
    if args.messages_only:
        print(json.dumps(build_messages(request), ensure_ascii=False, indent=2))
        return
    if args.output is None:
        parser.error('--output is required')
    if args.response_file and (args.endpoint or args.base_url):
        parser.error('--response-file cannot be combined with API URL options')
    if args.response_file:
        raw = args.response_file.read_text()
        transport = lambda messages: (raw, {'mode': 'offline_replay', 'network_calls': 0})
    else:
        settings = read_settings(args.env_file)
        model = args.model or settings.get('SCT_LLM_MODEL')
        key = settings.get('SCT_LLM_API_KEY')
        if not model:
            parser.error('Set SCT_LLM_MODEL in project .env or pass --model')
        if not key:
            parser.error('Set SCT_LLM_API_KEY in project .env or environment')
        endpoint = args.endpoint or chat_endpoint(args.base_url or settings.get('SCT_LLM_BASE_URL') or DEFAULT_BASE_URL)
        effort = args.reasoning_effort or settings.get('SCT_LLM_REASONING_EFFORT', '').strip() or DEFAULT_REASONING_EFFORT
        if effort not in REASONING_EFFORTS:
            parser.error('Invalid SCT_LLM_REASONING_EFFORT')
        transport = chat_transport(endpoint, model, key, json_mode=args.json_mode, reasoning_effort=effort)
    answer = OneShotScheduler(transport).schedule(request, args.output)
    print(json.dumps(answer, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
