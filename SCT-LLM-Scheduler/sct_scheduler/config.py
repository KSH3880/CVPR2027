"""Small literal .env reader: never executes shell syntax or prints secrets."""
import os
from pathlib import Path
from urllib.parse import urlparse
from .core import require

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = 'https://api.openai.com/v1'
DEFAULT_REASONING_EFFORT = 'medium'
REASONING_EFFORTS = ('none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max')


def read_settings(path=None):
    path = Path(path) if path is not None else PROJECT_ROOT / '.env'
    values = {}
    if path.exists():
        for number, line in enumerate(path.read_text().splitlines(), 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            require('=' in line, f'Invalid config entry at line {number}')
            key, value = line.split('=', 1)
            key, value = key.strip(), value.strip()
            require(key in {'SCT_LLM_BASE_URL', 'SCT_LLM_API_KEY', 'SCT_LLM_MODEL', 'SCT_LLM_REASONING_EFFORT'},
                    f'Unknown config field at line {number}')
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            require(key not in values, f'Duplicate config field at line {number}')
            values[key] = value
    for key in ('SCT_LLM_BASE_URL', 'SCT_LLM_API_KEY', 'SCT_LLM_MODEL', 'SCT_LLM_REASONING_EFFORT'):
        if key in os.environ:
            values[key] = os.environ[key]
    return values


def chat_endpoint(base_url):
    parsed = urlparse(base_url)
    require(bool(parsed.netloc) and not parsed.query and not parsed.fragment
            and parsed.username is None and parsed.password is None, 'Invalid base URL')
    require(not parsed.path.rstrip('/').endswith('/chat/completions'),
            'Use --endpoint for a full chat/completions URL')
    return base_url.rstrip('/') + '/chat/completions'
