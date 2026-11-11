import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from examples.make_synthetic_fixture import fixture
from sct_scheduler.core import build_messages, earliest_schedule, validate_request, validate_response
from sct_scheduler.client import loads, OneShotScheduler
from sct_scheduler.runtime import ReleaseClock
from sct_scheduler.recording import DemonstrationRecorder


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.request, self.response = fixture()

    def test_coupled_formula(self):
        validate_response(self.request, self.response)
        self.assertEqual(earliest_schedule(self.request['current_scene'], self.response['predictions']), {'A': 2., 'B': 0.})

    def test_enabling_formula(self):
        request, response = fixture('enabling')
        validate_response(request, response)
        self.assertEqual(earliest_schedule(request['current_scene'], response['predictions']), {'A': 0., 'B': 2.01})
        response['predictions'][1]['estimated_elapsed_time_s'] = 7.
        self.assertEqual(earliest_schedule(request['current_scene'], response['predictions']), {'A': 0., 'B': 0.})

    def test_joint_constraints(self):
        scene = self.request['current_scene']
        scene['initial_observation']['agents'].append({'agent_id': 'C'})
        scene['event_definitions']['C_ready'] = 'C detector'
        scene['event_agents'] = {'C_ready': 'C'}
        scene['coordination_constraints'].append({'type': 'enabling', 'enabling_event_id': 'B_ready', 'dependent_event_id': 'C_ready'})
        predictions = self.response['predictions'] + [{'event_id': 'C_ready', 'estimated_elapsed_time_s': 1.}]
        self.assertEqual(earliest_schedule(scene, predictions), {'A': 2., 'B': 0., 'C': 5.01})

    def test_enabling_simultaneous_event_rejected(self):
        request, response = fixture('enabling')
        response['schedule'][1]['release_time_s'] = 2.
        response['predicted_episode_events'][1]['episode_time_s'] = 5.
        with self.assertRaisesRegex(ValueError, 'violates constraints'):
            validate_response(request, response)

    def test_enabling_additional_wait_allowed(self):
        request, response = fixture('enabling')
        response['schedule'][1]['release_time_s'] = 8.
        response['predicted_episode_events'][1]['episode_time_s'] = 11.
        validate_response(request, response)

    def test_independent_agents_start_immediately(self):
        request,response=fixture()
        request['current_scene']['coordination_constraints']=[]
        for row in response['schedule']:row['release_time_s']=0.
        for row,pred in zip(response['predicted_episode_events'],response['predictions']):
            row['episode_time_s']=pred['estimated_elapsed_time_s']
        validate_response(request,response)
        response['schedule'][0]['release_time_s']=1.
        response['predicted_episode_events'][0]['episode_time_s']+=1.
        with self.assertRaisesRegex(ValueError,'Independent agents'):
            validate_response(request,response)

    def test_enabling_predecessor_still_fixed_at_zero(self):
        request, response = fixture('enabling')
        response['schedule'][0]['release_time_s'] = 1.
        response['schedule'][1]['release_time_s'] = 0.
        with self.assertRaisesRegex(ValueError, 'predecessor must start at zero'):
            validate_response(request, response)

    def test_enabling_equal_durations_requires_later_follower(self):
        request, response = fixture('enabling')
        response['predictions'][1]['estimated_elapsed_time_s'] = 5.
        self.assertEqual(earliest_schedule(request['current_scene'], response['predictions']), {'A': 0., 'B': .01})

    def test_infeasible_cycle(self):
        scene = self.request['current_scene']
        scene['event_definitions']['A_later'] = 'Later A event'
        scene['event_agents'] = {'A_later': 'A'}
        scene['coordination_constraints'].append({'type': 'enabling', 'enabling_event_id': 'A_later', 'dependent_event_id': 'B_ready'})
        predictions = self.response['predictions'] + [{'event_id': 'A_later', 'estimated_elapsed_time_s': 10.}]
        with self.assertRaisesRegex(ValueError, 'Infeasible'):
            earliest_schedule(scene, predictions)

    def test_request_contains_only_environment_and_initial_scene(self):
        messages = build_messages(self.request)
        payload = json.loads(messages[1]['content'])
        self.assertEqual(set(payload), {'environment_spec', 'current_scene'})
        self.assertEqual(payload, self.request)

    def test_legacy_demonstrations_rejected(self):
        self.request['demonstrations'] = []
        with self.assertRaisesRegex(ValueError, 'Invalid request keys'):
            build_messages(self.request)

    def test_current_future_data_rejected(self):
        self.request['current_scene']['initial_observation']['measured_events'] = []
        with self.assertRaises(ValueError):
            validate_request(self.request)

    def test_invalid_responses(self):
        for value in (float('nan'), float('inf'), -1, True, .001):
            with self.subTest(value=value):
                response = copy.deepcopy(self.response)
                response['schedule'][0]['release_time_s'] = value
                with self.assertRaises(ValueError):
                    validate_response(self.request, response)

    def test_missing_duplicate_or_extra_agent(self):
        for schedule in (self.response['schedule'][:1], self.response['schedule'] * 2,
                         self.response['schedule'] + [{'agent_id': 'C', 'release_time_s': 0}]):
            with self.subTest(schedule=schedule), self.assertRaises(ValueError):
                validate_response(self.request, dict(self.response, schedule=schedule))

    def test_arbitrary_wait_rejected(self):
        self.response['schedule'][0]['release_time_s'] += 1
        with self.assertRaises(ValueError):
            validate_response(self.request, self.response)

    def test_episode_arithmetic_rejected(self):
        self.response['predicted_episode_events'][0]['episode_time_s'] += .01
        with self.assertRaises(ValueError):
            validate_response(self.request, self.response)

    def test_legacy_example_citations_rejected(self):
        self.response['predictions'][0]['supporting_example_ids'] = ['invented']
        with self.assertRaises(ValueError):
            validate_response(self.request, self.response)

    def test_unknown_event_owner_rejected(self):
        self.response['predictions'][0]['agent_id'] = 'B'
        with self.assertRaises(ValueError):
            validate_response(self.request, self.response)

    def test_empty_schedule_rejected(self):
        self.response['schedule'] = []
        with self.assertRaises(ValueError):
            validate_response(self.request, self.response)

    def test_status_field_removed(self):
        for status in ('ok', 'infeasible', 'insufficient_information'):
            with self.subTest(status=status), self.assertRaisesRegex(ValueError, 'Invalid response keys'):
                validate_response(self.request, dict(self.response, status=status))

    def test_limitations_field_removed(self):
        validate_response(self.request, self.response)
        with self.assertRaisesRegex(ValueError, 'Invalid response keys'):
            validate_response(self.request, dict(self.response, limitations=[]))

    def test_strict_json(self):
        for text in ('{"x": 1, "x": 2}', '{"x": NaN}', '```json\n{}\n```'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                loads(text)

    def test_clock_quantization_and_one_way_release(self):
        clock = ReleaseClock({'A': .07, 'B': 0}, .03)
        self.assertEqual(clock.advance(0), {'A': True, 'B': False})
        self.assertEqual(clock.advance(2), {'A': True, 'B': False})
        self.assertEqual(clock.advance(3), {'A': False, 'B': False})
        clock.advance(4)
        self.assertEqual(len(clock.events), 2)
        self.assertAlmostEqual(clock.quantized_release_times()['A'], .09)
        with self.assertRaises(ValueError):
            clock.advance(0)

    def test_clock_decimal_boundary(self):
        clock = ReleaseClock({'A': .3, 'B': 0}, .1)
        self.assertFalse(clock.advance(3)['A'])

    def test_recorder_uses_detector_and_censors_missing(self):
        recorder = DemonstrationRecorder(self.request['current_scene'], 'history', {'A': 1, 'B': 0},
                                         ['episode_time_s', 'x'], ['episode_time_s', 'x'])
        recorder.sample(0, {'A': [-3], 'B': [-5]}, {})
        recorder.sample(6, {'A': [None], 'B': [-1]}, {})
        recorder.event('A_ready', 4.25)
        recorder.event('A_ready', 5)
        data = recorder.finish(6, {'success': False})
        self.assertEqual(data['measured_events'][0]['elapsed_since_release_s'], 3.25)
        self.assertFalse(data['measured_events'][1]['reached'])
        self.assertIsNone(data['measured_events'][1]['episode_time_s'])

    def test_one_shot_audit(self):
        base = Path(__file__).resolve().parents[1] / 'output_etc/check'
        base.mkdir(parents=True, exist_ok=True)
        calls = []
        def transport(messages):
            calls.append(messages)
            return json.dumps(self.response), {'model': 'fake'}
        with tempfile.TemporaryDirectory(dir=base) as tmp:
            scheduler = OneShotScheduler(transport)
            scheduler.schedule(self.request, Path(tmp) / 'episode')
            with self.assertRaises(ValueError):
                scheduler.schedule(self.request, Path(tmp) / 'again')
            self.assertEqual(len(calls), 1)
            self.assertTrue((Path(tmp) / 'episode/response.validated.json').is_file())

    def test_invalid_output_fails_without_retry(self):
        base = Path(__file__).resolve().parents[1] / 'output_etc/check'
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as tmp:
            scheduler = OneShotScheduler(lambda messages: ('bad JSON', {}))
            with self.assertRaises(ValueError):
                scheduler.schedule(self.request, Path(tmp) / 'episode')
            self.assertTrue((Path(tmp) / 'episode/failure.json').is_file())
            with self.assertRaises(ValueError):
                scheduler.schedule(self.request, Path(tmp) / 'retry')

    def test_gpu_allowlist_before_query(self):
        from sct_scheduler.isaac_adapter import check_gpu_mapping
        with patch('sct_scheduler.isaac_adapter.subprocess.check_output') as query:
            for gpu in (0, 5, 8, True):
                with self.assertRaises(ValueError):
                    check_gpu_mapping(gpu)
            query.assert_not_called()


if __name__ == '__main__':
    unittest.main()
