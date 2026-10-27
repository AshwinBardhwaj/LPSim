"""Verify portable source/input hashes; no Jetson binary requirement on A100."""
from pathlib import Path
import hashlib,json,sys

def verify(root):
    manifest=json.loads((root/'freeze.json').read_text());checked=0
    for group in ['source_hashes','input_hashes']:
        for relative,expected in manifest[group].items():
            p=root/relative
            if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:raise RuntimeError('Hash mismatch/missing: '+str(p))
            checked+=1
    print(f'PASS: {checked} frozen source/input hashes')
if __name__=='__main__':verify(Path(sys.argv[1]).resolve())
