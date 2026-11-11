import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from examples.make_synthetic_fixture import fixture
from sct_scheduler.__main__ import main


class InputCLITests(unittest.TestCase):
    def test_initial_request_messages_without_api_call(self):
        request,_ = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            scene=folder/'request.json'
            scene.write_text(json.dumps(request))
            output=io.StringIO()
            with patch('sys.argv',['scheduler',str(scene),'--messages-only']), \
                    patch('sct_scheduler.__main__.chat_transport') as transport, contextlib.redirect_stdout(output):
                main()
            messages=json.loads(output.getvalue())
            self.assertEqual(json.loads(messages[1]['content']),request)
            transport.assert_not_called()


if __name__ == '__main__':
    unittest.main()
