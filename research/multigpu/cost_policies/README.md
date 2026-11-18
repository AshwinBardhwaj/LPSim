# Cost-aware partition candidates — October 29, 2026

Two offline policy candidates are implemented through the same junction-owner-map contract as `local_scaling/policies.py`. The frozen single-GPU and multi-GPU backends are unchanged. No Modal job was launched.

| Candidate | Vertex balance | Cut-edge objective | Status |
|---|---|---|---|
| `work_balanced` | Expected road-vehicle residence | Known route visits | Implemented; uncalibrated proxy |
| `boundary_aware` | Same residence weights | Repeated ghost refresh + lane reservations + migration-rate proxy | Implemented; uncalibrated proxy |

The first isolates changing the balance weights; the second additionally changes communication weights. Both use deterministic METIS multilevel partitioning, seed 2710, target max/mean weighted load at most 1.05. The achieved balance is reported rather than assumed. All topology nodes remain assigned. Minimum integer weights retain unvisited topology; they are not measured costs.

Only complete route plans, road lengths/speeds/lanes, and an explicitly supplied expected simultaneous road population are used. Observed speed, position, queued state, departures, controller phases and future trajectories are not consulted. Residence is proportional to route visits times length/free-flow speed and normalized to the supplied population. This stationary approximation will be wrong for some queues and bursts; improving it is a later prediction experiment. It does not claim to estimate the actual 15-second evaluation trajectory.

For this backend, a road vehicle belongs to its destination junction and is ghosted to its source junction when the road is cut. Each step has two 432-byte full-record refreshes; each cut lane sends one 24-byte reservation. The migration component estimates ownership changes from route flow. Migration and ghost refresh may share a real message, so the sum is a selection proxy, not an exact forecast of wire bytes. The topology's reservation component does include unvisited cut roads.

`synchronized_step_ms` scores calibrated per-stage/per-rank costs as the **sum of the maximum rank duration at each barrier**. It is deliberately different from the maximum of each rank's total: different ranks can be slowest at different stages. It does not supply invented coefficients. `switch_decision` implements a conservative amortization gate, charging prediction, partitioning, migration and a total-horizon uncertainty allowance. This is a selection primitive, not an implemented online migration backend.

## Generated artifacts and checks

`artifacts/cost-policies-20261029/` contains eight candidate maps: Berkeley 8,192 and SF 98,304, two policies, two/three partitions. Each map was regenerated to verify determinism. `proxy-comparison.csv` compares all five policies with the same proxy, with measured runtime deliberately blank.

Seven contract tests pass. Four local Jetson runs, with two/three logical partitions and explicit messages, pass 80 total exact-reference timestep comparisons, including the backend's physical checks and ghost/authority checks. Their final physical hashes are identical. These gates cover Berkeley; the SF maps are structurally checked but have not yet been simulated. Physical multi-A100 correctness and performance tests remain outstanding for these new maps.

Illustrative **proxy values, not measurements**, for SF 98,304 / three partitions:

| Map | Predicted owned max/mean | Predicted ghost records | Boundary objective, byte-equivalents/step |
|---|---:|---:|---:|
| Paper balanced | 1.125 | 373.1 | 353,577 |
| Paper community | 1.432 | 215.1 | 198,090 |
| Work balanced | 1.020 | 822.4 | 718,129 |
| Boundary aware | 1.011 | 116.7 | 116,166 |

Balancing predicted work alone worsens the boundary objective here. That tradeoff justifies measuring the joint candidate; it is not evidence that the joint candidate is faster. Feature extraction plus partitioning cost about 0.34 seconds for Berkeley and 4.0 seconds for SF on this Jetson. Charge those costs for a cold run; separately report explicitly amortized reuse. A100-host construction timings remain unmeasured.

## Reproduction

From the repository root, using the preserved environment (NumPy and PyMetis 2023.1.1):

```bash
/tmp/lpsim-partition-env/bin/python -m unittest discover \
  -s research/multigpu/cost_policies -p 'test_*.py' -v
/tmp/lpsim-partition-env/bin/python research/multigpu/cost_policies/prepare.py \
  --output artifacts/cost-policies-new-run
/tmp/lpsim-partition-env/bin/python research/multigpu/cost_policies/validate.py \
  --artifacts artifacts/cost-policies-new-run \
  --binary /tmp/lpsim-local-scaling/local
```

The last command requires local GPU access. Rebuild that binary from the unchanged local-scaling source/frozen source closure if needed; see `../local_scaling/README.md`. It represents logical partitions on one device and must not be used for multi-GPU speedup claims. Output directories refuse overwrite. See `NEXT_STEPS.md` for the proposed calibration and research gates.
