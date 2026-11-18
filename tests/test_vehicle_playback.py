import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.export_vehicle_playback import export
from viz.process_routes import convert

class DynamicsTests(unittest.TestCase):
    def test_recorded_speed_and_acceleration_survive_sort_and_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p/'nodes.csv').write_text('index,x,y\n0,0,0\n1,.001,0\n')
            (p/'edges.csv').write_text('uniqueid,u,v,length\n10,0,1,100\n')
            (p/'routes.csv').write_text('0:[10]:0\n')
            trajectory=p/'trajectories.csv'
            trajectory.write_text('vehicle_id,timestamp,route_index,edge_id,pos_m,edge_id_kind,speed\n'
                                  '0,2,0,10,5,uniqueid,0\n0,0,0,10,0,uniqueid,2\n0,1,0,10,3,uniqueid,4\n')
            trips,_=convert(p,p/'routes.csv',trajectory)
            samples=trips[0]['samples']
            self.assertEqual([s['speed_mps'] for s in samples],[2,4,0])
            self.assertEqual([s['acceleration_mps2'] for s in samples],[None,2,-4])
            captured=[]
            convert(p,p/'routes.csv',trajectory,sink=captured.append)
            self.assertEqual(captured,trips)
            export(p,p/'routes.csv',trajectory,{'0'},p/'output')
            with (p/'output/observations.csv').open() as f:
                rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),3)
            self.assertEqual(rows[-1]['acceleration_mps2'],'-4.0')
            self.assertIn('longitude',rows[0])
            self.assertNotIn('/*TRAJECTORY_DATA*/null',(p/'output/index.html').read_text())
            self.assertIn('source_hashes',json.loads((p/'output/manifest.json').read_text()))

    def test_missing_speed_and_discontinuities_are_not_bridged(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            (p/'nodes.csv').write_text('index,x,y\n0,0,0\n1,.001,0\n')
            (p/'edges.csv').write_text('uniqueid,u,v,length\n10,0,1,100\n')
            (p/'routes.csv').write_text('0:[10]:0\n')
            source=p/'samples.csv'
            source.write_text('vehicle_id,timestamp,route_index,edge_id,pos_m,edge_id_kind,speed\n'
                              '0,0,0,10,0,uniqueid,2\n0,1,0,10,2,uniqueid,3\n'
                              '0,2,0,10,3,uniqueid,4\n0,2,0,10,4,uniqueid,4\n'
                              '0,3,0,10,5,uniqueid,8\n0,4,0,10,6,uniqueid,\n')
            trips,stats=convert(p,p/'routes.csv',source)
            self.assertEqual(stats['conflicting_timestamps'],1)
            self.assertEqual(len(trips),2)
            self.assertEqual([s['acceleration_mps2'] for s in trips[1]['samples']],[None,None])

if __name__=='__main__': unittest.main()
