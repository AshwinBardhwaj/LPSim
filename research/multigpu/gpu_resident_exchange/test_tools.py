import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('resident_prepare', HERE / 'prepare.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)
analysis_spec = importlib.util.spec_from_file_location('resident_analyze', HERE / 'analyze.py')
analyze = importlib.util.module_from_spec(analysis_spec)
analysis_spec.loader.exec_module(analyze)


class PreparationTests(unittest.TestCase):
    def test_paired_speedup_orientation(self):
        result = analyze.paired_speedup([2., 4., 6., 8., 10.], [1., 2., 3., 4., 5.], samples=100)
        self.assertEqual(result['median_speedup'], 2.)
        self.assertEqual(result['paired_repeat_interval_95'], [2., 2.])
        with self.assertRaises(ValueError):
            analyze.paired_speedup([1.] * 4, [1.] * 4)

    def test_adapters_preserve_frozen_physics(self):
        generated = prepare.adapters()
        compact = (prepare.FROZEN / 'tools/compact_motion.cuh').read_text()
        old = prepare.function(compact, 'advanceCompact')
        new = prepare.function(generated, 'advanceSlots')
        restored = new.replace('advanceSlots(const int* ids,', 'advanceCompact(').replace(
            'ids[j]<ids[chosenConnector]', 'j<chosenConnector')
        self.assertEqual(old, restored)
        self.assertIn('c.lanes[0],ids[i])', generated)
        self.assertIn('lookup[static_cast<unsigned int>(keys[k])]', generated)

    def test_source_drift_fails_closed(self):
        with self.assertRaises(ValueError):
            prepare.replace_once('old old', 'old', 'new')
        with self.assertRaises(ValueError):
            prepare.replace_once('missing', 'old', 'new')

    def test_freeze_hashes_and_reject_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'freeze'
            subprocess.run([sys.executable, str(HERE / 'prepare.py'), str(out)], check=True, capture_output=True)
            manifest = json.loads((out / 'manifest.json').read_text())
            self.assertIn('baseline/local.cu', manifest)
            self.assertIn('baseline/original.cu', manifest)
            for name, digest in manifest.items():
                self.assertEqual(hashlib.sha256((out / name).read_bytes()).hexdigest(), digest)
            again = subprocess.run([sys.executable, str(HERE / 'prepare.py'), str(out)], capture_output=True)
            self.assertNotEqual(again.returncode, 0)

    def test_logical_timing_is_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, str(HERE / 'evaluate.py'), '--build', tmp,
                '--model', 'missing', '--snapshot', 'missing', '--owners', 'missing', '--ranks', '3',
                '--output', str(Path(tmp) / 'out'), '--logical'], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('not scaling measurements', result.stderr)
            self.assertFalse((Path(tmp) / 'out').exists())

    def test_mutated_manifest_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); build = root / 'build'; build.mkdir()
            (build / 'manifest.json').write_text(json.dumps({'physics.cu': 'bad-digest'}))
            (build / 'physics.cu').write_text('modified')
            result = subprocess.run([sys.executable, str(HERE / 'evaluate.py'), '--build', str(build),
                '--model', 'missing', '--snapshot', 'missing', '--owners', 'missing', '--ranks', '3',
                '--output', str(root / 'out'), '--verify-only'], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Frozen source changed', result.stderr)


if __name__ == '__main__':
    unittest.main()
