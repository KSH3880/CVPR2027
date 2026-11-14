import copy
import json
import unittest
from pathlib import Path
from sct_scheduler.fewshot import attach_references, matching_initial_request
from sct_scheduler.core import build_messages
ROOT=Path(__file__).resolve().parents[1]
class FewshotTests(unittest.TestCase):
    def setUp(self):
        self.request=json.loads((ROOT/'output_etc/check/fewshot_single_scene_20261112_test01/request.json').read_text())
        self.request.pop('demonstrations')
        self.request['environment_spec']['controller']['timing_calibration']='Unavailable.'
    def test_baseline_preserved_and_prompt_selected(self):
        original=copy.deepcopy(self.request)
        actual=attach_references(self.request)
        self.assertEqual(self.request,original)
        self.assertEqual(actual['current_scene'],original['current_scene'])
        self.assertTrue(matching_initial_request(actual,original))
        self.assertEqual(build_messages(actual)[0]['content'],(ROOT/'prompts/fewshot/system.txt').read_text())
        self.assertEqual(build_messages(original)[0]['content'],(ROOT/'prompts/system.txt').read_text())
    def test_changed_scene_or_controller_rejected(self):
        actual=attach_references(self.request)
        actual['current_scene']['scene_id']='different'
        self.assertFalse(matching_initial_request(actual,self.request))
        actual=attach_references(self.request)
        actual['environment_spec']['controller']['description']='different'
        self.assertFalse(matching_initial_request(actual,self.request))
