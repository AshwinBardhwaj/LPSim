import csv
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('process_routes', Path(__file__).resolve().parents[1] / 'viz/process_routes.py')
m = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = m
SPEC.loader.exec_module(m)


class TrajectoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'nodes.csv').write_text('index,x,y\n0,0,0\n1,0.001,0\n2,0.00103,0\n3,0.00183,0\n')
        (self.root / 'edges.csv').write_text('uniqueid,u,v,length\n10,0,1,100\n20,1,2,3\n30,2,3,80\n')
        (self.root / 'routes').write_text('p:route:distance\n0:[10,20,30]:183\n')

    def run_rows(self, rows):
        with (self.root / 'samples').open('w') as f:
            w = csv.writer(f)
            w.writerow(['vehicle_id', 'timestamp', 'route_index', 'edge_id', 'pos_m', 'edge_id_kind'])
            w.writerows([[0, *r, 'uniqueid'] for r in rows])
        return m.convert(self.root, self.root / 'routes', self.root / 'samples')

    def test_sort_stops_and_skipped_connector(self):
        trips, stats = self.run_rows([(2, 2, 30, 8), (0, 0, 10, 98), (1, 0, 10, 98), (1, 0, 10, 98)])
        p = trips[0]['path']
        self.assertEqual(p[0][:2], p[1][:2])
        self.assertEqual([x[2] for x in p if x[2] in (0, 1, 2)], [0, 1, 2])
        self.assertAlmostEqual(p[-1][0], 0.00111)
        self.assertEqual(len(p), 5)  # includes both connector endpoints
        self.assertEqual(stats['duplicate_samples'], 1)

    def test_jump_splits(self):
        trips, stats = self.run_rows([(0, 0, 10, 0), (1, 0, 10, 1), (2, 2, 30, 70), (3, 2, 30, 71)])
        self.assertEqual(len(trips), 2)
        self.assertEqual(stats['kinematic_or_topology_splits'], 1)

    def test_conflicts_break_path(self):
        trips, stats = self.run_rows([(0, 0, 10, 0), (1, 0, 10, 1), (1, 0, 10, 2), (2, 0, 10, 3)])
        self.assertEqual(trips, [])
        self.assertEqual(stats['conflicting_timestamps'], 1)

    def test_overflow_discards_rest(self):
        trips, stats = self.run_rows([(0, 0, 10, 0), (1, 0, 10, 1), (2, 3, 30, 0), (3, 0, 10, 3)])
        self.assertEqual(len(trips[0]['path']), 2)
        self.assertEqual(stats['route_overflow'], 1)

    def test_reversed_curve_and_true_length(self):
        (self.root / 'edges.csv').write_text('uniqueid,u,v,length,geometry\n10,0,1,100,"LINESTRING (0.001 0, 0.0005 0.0005, 0 0)"\n20,1,2,3,\n30,2,3,80,\n')
        trips, _ = self.run_rows([(0, 0, 10, 40), (1, 0, 10, 60)])
        self.assertEqual(len(trips[0]['path']), 3)
        self.assertAlmostEqual(trips[0]['path'][1][1], 0.0005)

    def test_lane_offsets_and_disk_sort(self):
        (self.root / 'edges.csv').write_text('uniqueid,u,v,length,lanes\n10,0,1,100,2\n20,1,2,3,1\n30,2,3,80,1\n')
        (self.root / 'samples').write_text('vehicle_id,timestamp,route_index,edge_id,pos_m,lane_idx,edge_id_kind\n0,1,0,10,11,1,uniqueid\n0,0,0,10,10,0,uniqueid\n')
        trips, stats = m.convert(self.root, self.root/'routes', self.root/'samples', lane_width=3.2)
        self.assertGreater(trips[0]['path'][0][1], 0)
        self.assertLess(trips[0]['path'][1][1], 0)
        self.assertEqual(stats['recorded_lane_changes'], 1)
        self.assertEqual([s['lane'] for s in trips[0]['samples']], [0, 1])
        captured = []
        _, disk_stats = m.convert(self.root, self.root/'routes', self.root/'samples', lane_width=3.2, sink=captured.append)
        self.assertEqual(captured, trips)
        self.assertEqual(disk_stats, stats)

    def test_reverse_edges_offset_to_opposite_sides(self):
        (self.root / 'edges.csv').write_text('uniqueid,u,v,length,lanes\n10,0,1,100,2\n11,1,0,100,2\n')
        edges = m.load_edges(self.root)
        self.assertLess(edges[10].lane_point(50, 0)[1], 0)
        self.assertGreater(edges[11].lane_point(50, 0)[1], 0)

    def test_lane_out_of_range_is_excluded(self):
        (self.root / 'samples').write_text('vehicle_id,timestamp,route_index,edge_id,pos_m,lane_idx,edge_id_kind\n0,0,0,10,10,9,uniqueid\n')
        trips, stats = m.convert(self.root, self.root/'routes', self.root/'samples', lane_width=3.2)
        self.assertEqual(trips, [])
        self.assertEqual(stats['invalid_lane_samples'], 1)

    def test_legacy_rejected(self):
        (self.root / 'samples').write_text('vehicle_id,timestamp,edge_id,pos_m\n0,0,10,0\n')
        with self.assertRaisesRegex(ValueError, 'Legacy'):
            m.convert(self.root, self.root / 'routes', self.root / 'samples')


if __name__ == '__main__':
    unittest.main()
