"""One call per scene; injectable transport, no SDK/model dependency."""
import copy
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import urlparse
from .core import build_messages, validate_response, require


def loads(text):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key: ' + key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Non-finite JSON constant: ' + value)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def chat_transport(endpoint, model, api_key=None, timeout_s=120):
    """Explicit chat-completions-compatible endpoint; no retries or default model."""
    parsed = urlparse(endpoint)
    require(parsed.scheme == 'https' or (parsed.scheme == 'http' and parsed.hostname in ('localhost', '127.0.0.1', '::1')),
            'Use HTTPS except for loopback inference')
    require(bool(model), 'Explicit model required')
    def call(messages):
        payload = {'model': model, 'messages': messages, 'stream': False,
                   'response_format': {'type': 'json_object'}}
        headers = {'Content-Type': 'application/json'}
        if api_key:
            headers['Authorization'] = 'Bearer ' + api_key
        req = Request(endpoint, data=json.dumps(payload, allow_nan=False).encode(), headers=headers)
        with urlopen(req, timeout=timeout_s) as result:
            envelope = loads(result.read().decode())
        return envelope['choices'][0]['message']['content'], {
            'model': envelope.get('model', model), 'usage': envelope.get('usage'),
            'response_id': envelope.get('id'), 'finish_reason': envelope['choices'][0].get('finish_reason')}
    return call


class OneShotScheduler:
    def __init__(self, transport):
        self.transport = transport
        self.used_scenes = set()

    def schedule(self, request, output_dir):
        request = copy.deepcopy(request)
        messages = build_messages(request)
        scene_id = request['current_scene']['scene_id']
        require(scene_id not in self.used_scenes, 'Scene already submitted; no runtime replanning')
        destination = Path(output_dir).resolve()
        root = Path(__file__).resolve().parents[1] / 'output_etc/check'
        require(root.resolve() in destination.parents, 'Audit output must be under this project/output_etc/check/')
        destination.mkdir(parents=True, exist_ok=False)
        def save(name, value):
            (destination / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
        save('request.json', request)
        save('messages.json', messages)
        self.used_scenes.add(scene_id)
        start = time.monotonic()
        try:
            raw, metadata = self.transport(messages)
            (destination / 'response.raw.txt').write_text(raw)
            save('inference.json', dict(metadata, wall_latency_s=time.monotonic() - start,
                                        simulation_latency_s=0, calls=1))
            response = validate_response(request, loads(raw))
            save('response.validated.json', response)
            return response
        except Exception as exc:
            save('failure.json', {'error_type': type(exc).__name__, 'message': str(exc),
                                  'wall_latency_s': time.monotonic() - start, 'executable': False})
            raise
