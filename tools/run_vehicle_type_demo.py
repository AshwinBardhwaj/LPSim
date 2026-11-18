#!/usr/bin/env python3
"""Run two separated vehicles on the same corridor and export their dynamics."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.export_vehicle_playback import export, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output/'data').mkdir()
    network = ROOT/'data/networks/san_pablo_corridor_small'
    with (network/'route_pairs.csv').open() as f:
        pair = next(csv.DictReader(f))
    demand = output/'demand.csv'
    demand.write_text('dep_time,origin,destination\n'+''.join(
        f"{18000+120*i},{pair['origin']},{pair['destination']}\n" for i in range(2)))
    profiles = output/'vehicle_profiles.csv'
    profiles.write_text('vehicle_id,type_id,min_gap_m,time_headway_s,max_speed_mps,accel_mps2,decel_mps2\n'
                        '0,1,2,1.5,12,1.5,2.5\n1,2,5,2.5,6,0.8,1.5\n')
    config = output/'data/command_line_options.ini'
    config.write_text(f'''[General]
GUI=false
USE_CPU=false
NETWORK_PATH={network}/
USE_SP_ROUTING=true
USE_PREV_PATHS=false
ADD_RANDOM_PEOPLE=false
LIMIT_NUM_PEOPLE=2
NUM_PASSES=1
TIME_STEP=0.5
START_HR=5
END_HR=5.1666666667
OD_DEMAND_FILENAME={os.path.relpath(demand, network)}
SHOW_BENCHMARKS=true
REROUTE_INCREMENT=0
NUM_GPUS=1
''')
    binary=ROOT/'build/lpsim'
    env=dict(os.environ, LPSIM_SIGNAL_MODE='green', LPSIM_VEHICLE_PROFILES=str(profiles),
             LPSIM_TRAJECTORIES=str(output/'trajectories.csv'), OMP_NUM_THREADS='2')
    # Do not inherit unrelated capture destinations from another experiment.
    for key in ('LPSIM_SIGNALS', 'LPSIM_METRICS'):
        env.pop(key, None)
    with (output/'run.log').open('w') as log:
        subprocess.run([str(binary)], cwd=output, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    routes=next(output.glob('0_route*.csv'))
    export(network, routes, output/'trajectories.csv', {'0','1'}, output/'playback')
    with (output/'trajectories.csv').open() as f:
        rows=list(csv.DictReader(f))
    maxima={i:max(float(r['speed']) for r in rows if r['vehicle_id']==i) for i in ('0','1')}
    assert maxima['0'] > maxima['1'] + 1, maxima
    assert maxima['0'] <= 12.01 and maxima['1'] <= 6.01, maxima
    assert {r['vehicle_type'] for r in rows}=={'1','2'}
    inputs=[binary,profiles,config,demand,network/'nodes.csv',network/'edges.csv']
    (output/'manifest.json').write_text(json.dumps(dict(
        command=sys.argv, signal_mode='green', vehicle_count=2, max_speed_mps=maxima,
        inputs={str(p):digest(p) for p in inputs},
        note='Illustrative parameters, not calibrated sedan/truck measurements. Two separated departures, no profiling.'),indent=2))
    print('GPU smoke test passed:',maxima)


if __name__=='__main__': main()
