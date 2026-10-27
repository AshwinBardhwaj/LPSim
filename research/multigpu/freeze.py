"""Snapshot the exact source closure and validated input/output hashes."""
from pathlib import Path
import json, subprocess, shutil, sys
from prepare import ROOT, source_closure, sha

def freeze(root):
    root=root.resolve();target=root/'source';target.mkdir(exist_ok=False)
    starts=[ROOT/'research/multigpu/partitioned.cu',ROOT/'research/multigpu/hardware_probe.cu',ROOT/'tools/incremental_simulator.cu']
    files=source_closure(starts)|{p for p in (ROOT/'research/multigpu').iterdir() if p.is_file()}
    for p in sorted(files):
        q=target/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    def capture(args):
        p=subprocess.run(args,cwd=ROOT,text=True,capture_output=True);return dict(argv=args,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
    manifest=dict(schema=1,date='2026-10-27',git_commit=capture(['git','rev-parse','HEAD'])['stdout'].strip(),git_status=capture(['git','status','--short'])['stdout'],source_authority='Self-contained source snapshot and hashes; unrelated dirty working-tree files excluded.',reference='compact SoA, radix leaders, indexed parallel following junctions; incremental separately selectable',dt=.25,horizon=600,controller_mode=0,block_threads=128,seed_reference=1901,build_commands=['bash source/research/multigpu/build.sh bin sm_87','bash source/research/multigpu/build.sh bin sm_80'],environment={'LPSIM_COMPACT_MOTION':'1','LPSIM_LEADER_INDEX':'1','LPSIM_LEADER_SORT':'radix','LPSIM_MOTION_CONTEXT':'3','LPSIM_JUNCTION_PARALLEL':'1','LPSIM_JUNCTION_REQUEST_INDEX':'1','LPSIM_JUNCTION_EXCLUSIVE':'0'},compiler=capture(['/usr/local/cuda/bin/nvcc','--version']),host=capture(['uname','-a']))
    for kind,folder in [('source_hashes',target),('input_hashes',root/'inputs'),('validation_hashes',root/'validation'),('binary_hashes',root/'bin')]:manifest[kind]={str(p.relative_to(root)):sha(p) for p in sorted(folder.rglob('*')) if p.is_file()}
    manifest['input_hashes']['workloads.json']=sha(root/'workloads.json')
    manifest['hardware_hashes']={str(p.relative_to(root)):sha(p) for p in sorted((root/'hardware').rglob('*')) if p.is_file()}
    manifest['standalone_reference_hashes']={str(p.relative_to(root)):sha(p) for name in ['standalone-reference','standalone-incremental'] for p in sorted((root/name).rglob('*')) if p.is_file()}
    manifest['power_record']='hardware/jetson-power.txt';manifest['validation_gate']=json.loads((root/'validation_gate.json').read_text())
    manifest['standalone_validation']=json.loads((root/'reference_validation.json').read_text())
    (root/'freeze.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (root/'freeze.sha256').write_text(sha(root/'freeze.json')+'  freeze.json\n')
if __name__=='__main__':freeze(Path(sys.argv[1]))
