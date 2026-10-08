import importlib.util
import json
import csv
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('prepare_playback', Path(__file__).resolve().parents[1] / 'viz/prepare_playback.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class PlaybackTests(unittest.TestCase):
    def test_boundaries_preserve_samples_and_stops(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / 'trips.json'
            path = [[0, 0, 299], [0, 0, 300], [0.1, 0, 301], [0.2, 0, 602]]
            source.write_text(json.dumps([{'vendor': '0', 'path': path}]))
            manifest = m.prepare(source)
            windows = [json.loads((source.parent / 'playback' / f).read_text()) for f in manifest['chunks'].values()]
            actual = {tuple(p) for w in windows for t in w for p in t['path']}
            self.assertEqual(actual, set(map(tuple, path)))
            self.assertEqual(manifest['start'], 299)
            self.assertEqual(manifest['end'], 602)
            self.assertTrue(all(len(w[0]['path']) >= 2 for w in windows))

    def test_lane_events_once_across_window_boundaries(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / 'trips.json'
            source.write_text(json.dumps([{'vendor':'7', 'path':[[0,0,299],[0,0,300],[0,0,301]],
                'samples':[{'edge':10,'route_index':0,'lane':0}, None, {'edge':10,'route_index':0,'lane':1}]}]))
            manifest=m.prepare(source)
            with (source.parent/'playback'/manifest['lane_changes']).open() as f:
                events=list(csv.DictReader(f))
            self.assertEqual(len(events),1)
            self.assertEqual(events[0]['lane_after'],'1')
            self.assertEqual(events[0]['time_before'],'299')

    def test_stream_large_object(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'trips.json'
            trip = {'vendor': 'long', 'path': [[0, 0, i] for i in range(100000)]}
            p.write_text(json.dumps([trip, {'vendor': 'next', 'path': []}]))
            self.assertEqual(list(m.iter_trips(p))[0], trip)

    def test_truncation_is_error(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'trips.json'
            p.write_text('[{"vendor":"x"}')
            with self.assertRaises(ValueError):
                list(m.iter_trips(p))
