import unittest
from sct_scheduler.executable import validate_executable


class ExecutableTests(unittest.TestCase):
    def setUp(self):
        self.request={'current_scene':{'initial_observation':{'agents':[
            {'agent_id':'A'}, {'agent_id':'B'}]}}}

    def test_prediction_errors_do_not_change_releases(self):
        answer={'schedule':[{'agent_id':'A','release_time_s':3.21},
                            {'agent_id':'B','release_time_s':9.87}],
                'predictions':[], 'predicted_episode_events':[]}
        self.assertIs(validate_executable(self.request,answer),answer)
        self.assertEqual([x['release_time_s'] for x in answer['schedule']],[3.21,9.87])

    def test_missing_or_ambiguous_release_is_not_executable(self):
        for rows in ([{'agent_id':'A','release_time_s':0}],
                     [{'agent_id':'A','release_time_s':0}]*2):
            with self.assertRaises(ValueError):
                validate_executable(self.request,{'schedule':rows})

    def test_invalid_time_rejected(self):
        for time in (-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):
                validate_executable(self.request,{'schedule':[
                    {'agent_id':'A','release_time_s':time},
                    {'agent_id':'B','release_time_s':0}]})
