import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from generate_run_report import aggregate, split_interval
from record_run_metrics import parse_tegrastats
from process_routes import Edge


class ReportTests(unittest.TestCase):
    def test_distance_time_conservation_across_short_edge(self):
        edges={10:Edge(0,1,100,[],[]),20:Edge(1,2,3,[],[]),30:Edge(2,3,80,[],[])}
        parts=split_interval([10,20,30],[0,100,103,183],edges,(0,0,10,98,0),(1,2,30,8,0))
        self.assertAlmostEqual(sum(p[3] for p in parts),13)
        self.assertAlmostEqual(sum(p[2]-p[1] for p in parts),1)
        self.assertEqual([p[0] for p in parts],[10,20,30])

    def test_stopped_interval(self):
        edges={10:Edge(0,1,100,[],[])}
        self.assertEqual(split_interval([10],[0,100],edges,(0,0,10,5,0),(2,0,10,5,0)),[(10,0,2,0)])

    def test_telemetry_units(self):
        r=parse_tegrastats('RAM 2000/8000MB GR3D_FREQ 75%@[600,600] GPU@48.5C VDD_IN 12500mW/11000mW')
        self.assertEqual(r['power_w'],12.5)
        self.assertEqual(r['gpu_util_pct'],75)
        self.assertNotIn('gpu_util_pct',parse_tegrastats('GR3D_FREQ @600'))

    def test_aggregation_sort_duplicates_and_window_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            (p/'nodes.csv').write_text('index,x,y\n0,0,0\n1,0.001,0\n')
            (p/'edges.csv').write_text('uniqueid,u,v,length,lanes,speed_mph\n10,0,1,100,1,25\n')
            (p/'routes.csv').write_text('p:route:distance\n0:[10]:100\n')
            (p/'samples.csv').write_text('vehicle_id,timestamp,route_index,edge_id,pos_m,speed,edge_id_kind\n0,301,0,10,20,10,uniqueid\n0,299,0,10,0,10,uniqueid\n0,301,0,10,20,10,uniqueid\n')
            v,s,b,e,summary=aggregate(p,p/'routes.csv',p/'samples.csv',p)
            self.assertAlmostEqual(summary['observed_vmt']*1609.344,20)
            self.assertAlmostEqual(summary['observed_vht']*3600,2)
            self.assertEqual(len(b),2)
            self.assertEqual(summary['quality']['duplicates'],1)
            self.assertAlmostEqual(sum(x['observed_vht'] for x in b),summary['observed_vht'])


class AutomaticReportTests(unittest.TestCase):
    def test_recorder_generates_report_after_success(self):
        import subprocess
        import json
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            inputs=root/'run'/'inputs'
            network=inputs/'network'
            network.mkdir(parents=True)
            (network/'nodes.csv').write_text('index,x,y\n0,0,0\n1,0.001,0\n')
            (network/'edges.csv').write_text('uniqueid,u,v,length,lanes,speed_mph\n10,0,1,100,1,25\n')
            (inputs/'routes.csv').write_text('p:route:distance\n0:[10]:100\n')
            (inputs/'trajectories.csv').write_text('vehicle_id,timestamp,route_index,edge_id,pos_m,speed,edge_id_kind\n0,0,0,10,0,10,uniqueid\n0,2,0,10,20,10,uniqueid\n')
            result=subprocess.run([sys.executable,str(ROOT/'tools/record_run_metrics.py'),
                '--output',str(root/'capture'),'--no-telemetry',
                '--report-run-dir',str(root/'run'),'--',sys.executable,'-c','pass'],
                capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            report=root/'capture'/'report'
            for name in ('report.html','report.pdf','road_maps.png','traffic_over_time.png'):
                self.assertGreater((report/name).stat().st_size,0)
            manifest=json.loads((report/'manifest.json').read_text())
            self.assertIn('whole_command_runtime',manifest['compute_availability'])
            self.assertAlmostEqual(manifest['summary']['observed_vht'],2/3600)
