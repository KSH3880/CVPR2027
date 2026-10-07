import unittest
from carry_planner.physical_contact import classify_contacts

class ContactTest(unittest.TestCase):
    def test_expected_and_zero_force_contacts_excluded(self):
        owners={0:('agent',0),1:('agent',0),2:('agent',1),3:('box',0),4:('box',1)}
        contacts=[dict(body0=a,body1=b,**{'lambda':f}) for a,b,f in
                  ((0,-1,100),(0,1,5),(0,3,5),(0,2,0))]
        self.assertFalse(classify_contacts(contacts,owners)['total'])

    def test_pair_types_and_threshold(self):
        owners={0:('agent',0),1:('agent',1),2:('box',0),3:('box',1)}
        contacts=[dict(body0=a,body1=b,**{'lambda':f}) for a,b,f in
                  ((0,1,2),(0,3,3),(2,3,.5))]
        hits=classify_contacts(contacts,owners,1)
        self.assertTrue(hits['agent_agent']); self.assertTrue(hits['agent_box'])
        self.assertFalse(hits['box_box'])
