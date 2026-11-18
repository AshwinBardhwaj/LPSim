"""Generate PI talking points from measured output; no invented speedups."""
from pathlib import Path
import json,pandas as pd
r=Path('reports/a100-study-20261028');s=json.loads((r/'findings.json').read_text());t=pd.read_csv(r/'timing-summary.csv');d=pd.read_csv(r/'diagnostic-windows.csv');a=pd.read_csv(r/'allocations.csv')
means=t.groupby('label').median_ms.mean()/1000
best=t[t.label!='single'].groupby('label').median_ms.mean().idxmin()
script=f'''# PI presentation script — October 28, 2026

## Opening: the hypothesis (45 seconds)

“My research question is whether information about a traffic network and its expected demand can improve the organization of GPU work. The single-GPU experiments taught us that congestion is not automatically recoverable scheduling cost. We have now moved to a setting where partition boundaries introduce a concrete cost: communicating vehicle state and preserving intersection dependencies across devices.

Today I am presenting a correctness and baseline study, not claiming that we have already built a successful learned allocator.”

## Why correctness came first (1 minute)

“An allocator is only useful if it preserves the simulation. We first compared logical partitions with physical A100 devices, then implemented concurrent execution and direct peer transfers. The original physical test passed 720 timestep comparisons. The concurrent backend passed another 7,920 comparisons, including full 600-second bursty runs under all three policies. We checked exact vehicle records, ownership, ghost consistency, gaps and conflicting junction movements.

The new decision interface assigns complete junctions and their incoming roads to GPUs. Vehicles migrate when their authoritative region changes. Boundary ghosts and receiving-space reservations preserve the same traffic calculations.”

Show: report correctness and protocol pages.

## Why these baselines (1 minute)

“We need to beat inexpensive policies before fitting a prior. Geography keeps nearby roads together. Vehicle-count balancing distributes observed active vehicles. The graph heuristic trades a static communication proxy against population balance. All three execute the same kernels and use the same indexes.

We use {s['cases']} development workloads spanning requested populations and departure patterns. Counts alone can be misleading because requested vehicles may remain pending. We therefore record pending, stopped, moving, connector and completed populations separately.

To make count balancing meaningful, all policies are built from the same observed state at 180 seconds. They remain fixed through 600 seconds. Our repeated performance result is this 420-second continuation, not a repeated full 600-second deployment measurement.”

Show: Figure 1, runtime table, allocation metadata.

## What the timings show (1 minute)

“Across the measured cases, three-GPU runtime ranges from {s['peer_slowdown_min']:.2f} to {s['peer_slowdown_max']:.2f} times the single-GPU runtime. The mean continuation times are {means.get('single',float('nan')):.3f} seconds on one GPU, {means.get('geography',float('nan')):.3f} for geography, {means.get('vehicle_count',float('nan')):.3f} for active count and {means.get('communication_graph',float('nan')):.3f} for the graph heuristic. These averages summarize different workloads; the per-case table and variation matter more than a single headline.

The best average three-GPU policy in this development set is {best}. That is not a trained prediction result. We charge its assignment construction and show cold setup costs separately.”

Show: Figure 1 and cold-cost table. Pause to distinguish a policy improvement from speedup over one GPU.

## What the allocation measurements explain (1 minute)

“The point of the experiment is to identify a decision a prior can improve. Figure 3 shows how partition choice changes actual boundary bytes, and Figure 5 shows the active populations on the GPUs over time. A count-balanced partition can scatter connected junctions and increase communication; a geographic partition can keep communication local while accepting a population imbalance.

Figure 2 separates motion/index work, junction work, spawning and exchange stages. Exchange includes packing, count handling, transfers, waiting and applying received state. It must not be described as raw NVLink latency.

We also retain N stable-ID slots on every GPU. Thus dividing ownership does not divide every array scan or index construction by three. That is an important baseline limitation, not something a learned prior can predict away.”

Show: Figures 2–5.

## How to read the Nsight results (1 minute)

“The Nsight tables report actual kernel counters: long-scoreboard stalls, SM throughput, achieved warp occupancy, eligible warps and active threads per instruction. They are sampled separately at matched traffic states so profiling does not contaminate timing repetitions.

Long-scoreboard means a warp is waiting on a memory dependency; it does not mean the memory bus is saturated or that a communication transfer is responsible. Achieved occupancy describes active warps during the kernel. It does not measure CPU-side waits between kernels. We need both kernel counters and complete-step timings to understand why the GPUs finish when they do.

Across the sampled cases, motion achieved roughly 6% occupancy, 59–62% long-scoreboard stalls and about 2% SM throughput. Active threads per instruction fell from about 26 on one GPU to about 12 in the partitioned runs. Junction arbitration showed about 82–83% long-scoreboard stalls, but its individual launches lasted only about 20–24 microseconds. These counters diagnose kernel behavior; the complete-step breakdown shows that exchange and coordination are the larger opportunity.

The profiler allocation’s actual GPU model is recorded separately. We do not silently equate 40 GB and 80 GB A100 measurements.”

Show: Figure 6 and kernel tables; cite the saved .ncu-rep files and nsight-summary.csv.

## The decision gate and next steps (1 minute)

“Our optimistic balance-only screen asks how much time could be removed if the measured motion and junction stages were perfectly balanced, leaving communication unchanged. Its largest sampled saving is {s['balance_screen_max_percent']:.2f}% of step time. This is not an implemented oracle partition and does not bound improvements from reducing communication or changing the baseline.

We also tested a restricted measured selector: choose among the three existing policies using the excluded warm-up, then evaluate on five separate repeats. It selected geography in every completed case. That adds no benefit over simply choosing geography, and charging calibration makes it worse. It is not an oracle over all possible partitions.

The next decision should follow the measured dominant costs. If dense replicated work and phase coordination dominate, we should first establish an efficient partition-local baseline. Then we can measure regional compute and boundary costs, construct a cost-informed oracle on independent timing repetitions, and ask whether topology and expected route flows can predict a partition that retains those savings.

A prior only earns its place if it beats the strongest cheap policy on unseen demand, including construction and migration costs. Our preregistered target remains a 10% geometric-mean complete-runtime improvement with correctness and regression constraints.”

## Closing (20 seconds)

“We have made the multi-GPU mechanism concrete and validated it. The value of this study is that we can now separate traffic physics, partition decisions, communication and implementation overhead using measured data. The study currently has ${s['posted_study_cost_usd']:.3f} in posted Modal usage against the $10 limit. I would use the next experiment to resolve the dominant measured bottleneck, rather than train a predictor before there is an opportunity for it to recover.”

## Short answers to likely questions

- **Is this calibrated Berkeley demand?** No. It is frozen synthetic route demand with a modest lane/speed preference and randomized departure patterns.
- **Did three GPUs improve performance?** Use the per-case ratios in relative-runtime.csv. Do not confuse lower communication with lower total runtime.
- **Is the graph policy a network prior?** It is a topology-based heuristic baseline; it does not forecast future demand or learn from trajectories.
- **Did we prove a traffic-aware allocator cannot work?** No. We measured this ownership/transport implementation and these workloads. Replicated work and barriers constrain what partition changes can recover.
- **Have we tested generalization?** No learned selector was trained; validation/test inputs remain reserved. All these performance cases are development cases on Berkeley.
- **What counts as publishable evidence?** A measured oracle opportunity, a predictor trained only on allowed information, robust gains over strong baselines on held-out demand/networks, and full accounting of correctness and overhead.
'''
(r/'presentation-script.md').write_text(script)
(r/'talking-points.md').write_text(f'''# PI talking points

- Research target: network/demand information that reduces measured computation and boundary coordination costs.
- Correctness: 720 original logical/physical + 7,920 concurrent timestep comparisons passed.
- Fairness: identical physics/indexes, current-state assignments at 180 s, five repetitions; diagnostics/profiling separate.
- Coverage: {s['cases']} development workloads; requested and actual active populations distinguished.
- Runtime: three-GPU/single-GPU ratio {s['peer_slowdown_min']:.2f}–{s['peer_slowdown_max']:.2f}; see exact case table.
- Allocation metrics: active imbalance, ghosts, cut roads/lanes, state/reservation bytes, ownership transfers, construction/setup costs, per-stage critical path.
- Counters: long-scoreboard, SM throughput, occupancy, eligible warps and active threads; per-kernel metrics are not wall-time utilization.
- Architecture limitation: dense replicated slots/index work and conservative barriers remain.
- Oracle screen: at most {s['balance_screen_max_percent']:.2f}% in the sampled balance-only calculation; not an executed cost-oracle partition.
- Next: resolve measured baseline bottlenecks, then cost-oracle gate, then evaluate a traffic prior on held-out data.
- Posted study usage: ${s['posted_study_cost_usd']:.3f}/$10; preserve all raw artifacts and report negative findings.
''')
# Printable speaker notes accompany the editable Markdown.
import textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
lines=[]
for paragraph in script.splitlines():
 heading=paragraph.startswith('#');paragraph=paragraph.lstrip('# ').replace('**','').replace('$','USD ')
 if heading:lines.append(('',False))
 lines.extend((line,heading) for line in (textwrap.wrap(paragraph,88 if not heading else 72,break_long_words=False) or ['']))
with PdfPages(r/'presentation-script.pdf') as pdf:
 for start in range(0,len(lines),45):
  fig=plt.figure(figsize=(8.3,11.7));y=.94
  for line,heading in lines[start:start+45]:
   fig.text(.07,y,line,fontsize=11 if not heading else 12,weight='bold' if heading else 'normal',va='top');y-=.019
  fig.text(.07,.035,f'LPSim A100 study • Speaker notes • {start//45+1}',fontsize=9,color='#555');pdf.savefig(fig);plt.close(fig)
