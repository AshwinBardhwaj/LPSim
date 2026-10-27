"""CPU tests of policy/input semantics, independent of GPU performance."""
from pathlib import Path
import copy, json, tempfile, unittest
from prepare import CAR, ROOT, partition, read_snapshot, POLICIES, sha

class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.geometry={'roads':[dict(id=i,a=i,b=i+1,start=[i*100,0],end=[(i+1)*100,0],lanes=1,length=100,speed=10) for i in range(8)]}
    def car(self,edge,state=1):
        c=[0]*107;c[0]=state;c[2]=2;c[6]=edge;return tuple(c)
    def test_pending_does_not_count_as_active(self):
        cars=[self.car(0,0)]*100+[self.car(7,3)]*50
        _,meta=partition(self.geometry,cars,'vehicle_count')
        self.assertEqual(meta['population_weights'],'unit_junction_fallback')
    def test_active_count_uses_only_active(self):
        cars=[self.car(0)]*7+[self.car(3,2)]*3+[self.car(0,0)]*100
        _,meta=partition(self.geometry,cars,'vehicle_count')
        self.assertEqual(sum(meta['loads']),10)
        self.assertEqual(meta['population_weights'],'observed_road_and_connector')
    def test_determinism_and_complete_ownership(self):
        cars=[self.car(i) for i in range(8)]*3
        for policy in POLICIES:
            a,m=partition(self.geometry,cars,policy);b,n=partition(self.geometry,copy.deepcopy(cars),policy)
            self.assertEqual(a,b);self.assertEqual(m,n);self.assertEqual(len(a),9);self.assertEqual(set(a),{0,1,2});self.assertEqual(sum(m['loads']),24)
    def test_snapshot_rejects_truncation(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'snapshot.bin';p.write_bytes(b'\0'*8)
            with self.assertRaises(ValueError):read_snapshot(p)
    def test_unknown_policy_rejected(self):
        with self.assertRaises(ValueError):partition(self.geometry,[],'not-a-policy')

if __name__=='__main__':unittest.main()
