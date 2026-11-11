import io
import contextlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

from sct_scheduler.client import chat_transport
from sct_scheduler.config import chat_endpoint, read_settings, DEFAULT_BASE_URL


class ApiTests(unittest.TestCase):
    def test_base_url_keeps_gateway_path(self):
        expected = 'https://api.baze.co/v1/gateway/chat/completions'
        self.assertEqual(chat_endpoint('https://api.baze.co/v1/gateway'), expected)
        self.assertEqual(chat_endpoint('https://api.baze.co/v1/gateway/'), expected)
        self.assertEqual(chat_endpoint(DEFAULT_BASE_URL), 'https://api.openai.com/v1/chat/completions')

    def test_literal_config_and_environment_precedence(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            path = Path(tmp) / '.env'
            path.write_text('SCT_LLM_API_KEY="test-key"\nSCT_LLM_MODEL=$(not-executed)\n')
            self.assertEqual(read_settings(path)['SCT_LLM_API_KEY'], 'test-key')
            self.assertEqual(read_settings(path)['SCT_LLM_MODEL'], '$(not-executed)')
            os.environ['SCT_LLM_MODEL'] = 'test-model'
            self.assertEqual(read_settings(path)['SCT_LLM_MODEL'], 'test-model')

    def test_request_headers_body_and_metadata(self):
        envelope = {'id': 'test-id', 'model': 'test-model', 'usage': {'total_tokens': 100},
                    'choices': [{'message': {'content': '{"status":"ok"}'}, 'finish_reason': 'stop'}]}
        for mode in (False, True):
            with self.subTest(json_mode=mode), patch('sct_scheduler.client.urlopen') as call:
                call.return_value.__enter__.return_value.read.return_value = json.dumps(envelope).encode()
                transport = chat_transport(chat_endpoint(DEFAULT_BASE_URL), 'test-model', 'test-secret', json_mode=mode, reasoning_effort='medium')
                messages = [{'role': 'user', 'content': 'test'}]
                raw, metadata = transport(messages)
                req = call.call_args[0][0]
                self.assertEqual(req.full_url, DEFAULT_BASE_URL + '/chat/completions')
                self.assertEqual(req.get_method(), 'POST')
                self.assertEqual(req.get_header('Authorization'), 'Bearer test-secret')
                payload = json.loads(req.data)
                self.assertEqual(payload['messages'], messages)
                self.assertEqual(payload['reasoning_effort'], 'medium')
                self.assertEqual(metadata['reasoning_effort'], 'medium')
                self.assertEqual('response_format' in payload, mode)
                self.assertNotIn('test-secret', raw + json.dumps(metadata) + req.data.decode())
                call.assert_called_once()

    def test_http_error_does_not_echo_key_or_retry(self):
        with patch('sct_scheduler.client.urlopen') as call:
            call.side_effect = HTTPError(DEFAULT_BASE_URL, 401, 'test-secret', {}, io.BytesIO(b'test-secret'))
            with self.assertRaisesRegex(RuntimeError, 'HTTP 401') as result:
                chat_transport(chat_endpoint(DEFAULT_BASE_URL), 'test-model', 'test-secret')([])
            self.assertNotIn('test-secret', str(result.exception))
            call.assert_called_once()

    def test_invalid_urls(self):
        for url in ('https://name:secret@example.com', 'https://example.com?key=secret', 'https://example.com/chat/completions'):
            with self.subTest(url=url), self.assertRaises(ValueError):
                chat_endpoint(url)

    def test_cli_uses_env_file_and_default_gateway(self):
        from sct_scheduler.__main__ import main
        fixture = Path(__file__).resolve().parents[1] / 'examples/synthetic_coupled_request.json'
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            settings = Path(tmp) / '.env'
            settings.write_text('SCT_LLM_API_KEY=test-secret\nSCT_LLM_MODEL=test-model\n')
            argv = ['scheduler', str(fixture), '--env-file', str(settings), '--output', 'unused']
            with patch('sys.argv', argv), patch('sct_scheduler.__main__.chat_transport') as transport, \
                    patch('sct_scheduler.__main__.OneShotScheduler') as scheduler, \
                    contextlib.redirect_stdout(io.StringIO()):
                scheduler.return_value.schedule.return_value = {'status': 'ok'}
                main()
                transport.assert_called_once_with(DEFAULT_BASE_URL + '/chat/completions',
                                                   'test-model', 'test-secret', json_mode=False, reasoning_effort='medium')

    def test_effort_blank_fallback_and_cli_override(self):
        from sct_scheduler.__main__ import main
        fixture = Path(__file__).resolve().parents[1] / 'examples/synthetic_coupled_request.json'
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            settings = Path(tmp) / '.env'
            settings.write_text('SCT_LLM_API_KEY=test-secret\nSCT_LLM_MODEL=test-model\nSCT_LLM_REASONING_EFFORT=\n')
            for override, expected in (([], 'medium'), (['--reasoning-effort', 'high'], 'high')):
                argv = ['scheduler', str(fixture), '--env-file', str(settings), '--output', 'unused'] + override
                with patch('sys.argv', argv), patch('sct_scheduler.__main__.chat_transport') as transport, \
                        patch('sct_scheduler.__main__.OneShotScheduler') as scheduler, \
                        contextlib.redirect_stdout(io.StringIO()):
                    scheduler.return_value.schedule.return_value = {'status': 'ok'}
                    main()
                    self.assertEqual(transport.call_args.kwargs['reasoning_effort'], expected)

    def test_invalid_effort_rejected_before_network(self):
        with patch('sct_scheduler.client.urlopen') as network:
            with self.assertRaisesRegex(ValueError, 'Invalid reasoning effort'):
                chat_transport(chat_endpoint(DEFAULT_BASE_URL), 'test-model', reasoning_effort='typo')
            network.assert_not_called()


if __name__ == '__main__':
    unittest.main()
