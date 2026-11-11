import unittest
from examples.make_synthetic_fixture import fixture
from sct_scheduler.stage1 import policy_manifest, require_stage1_request


class Stage1IdentityTests(unittest.TestCase):
    def test_old_controller_schedule_cannot_run_as_stage1(self):
        request, _ = fixture()
        request['current_scene']['controller_id'] = 'sha256:c2083707d13de3f32e9244bf0184a7a5b7e8a4d580cbb65a40308d387dd1705b'
        with self.assertRaisesRegex(ValueError, 'controller mismatch'):
            require_stage1_request(request)

    def test_stage1_request_identity(self):
        request, _ = fixture()
        request['current_scene']['controller_id'] = policy_manifest()['controller_id']
        manifest = require_stage1_request(request)
        self.assertEqual(manifest['epoch'], 26000)
        self.assertFalse(manifest['stage2_coordination'])
