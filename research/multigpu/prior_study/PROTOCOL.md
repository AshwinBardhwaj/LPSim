# Frozen protocol, before new timing results

Backend: full-rebuild exchange mode 3; unchanged physics, dt 0.25 s,
72 s protected controller, 600 s horizon. Three physical A100 80 GB GPUs.
Berkeley only. Synthetic arterial-biased routes, not calibrated OD data.
Four patterns: distributed, concentrated, mixed, bursty. Requested counts
4096 and 16384; calibration seed 3301, held-out seed 3302. All begin pending.
Report actual road/connector/pending/completed populations at 30 s intervals.

Candidates: geography, paper_balanced, boundary_aware. Ownership stays fixed
within a run. Include startup and map generation; normal ownership transfers
are already timed. Repartition migration cost is zero because no repartition
occurs. Do not claim an online selector or external-network generalization.
Three shuffled uninstrumented repetitions; separate sampled diagnostic runs.
Exact physical final-state equality across every policy/repetition; sampled
physical and ghost checks; short every-step single-device reference gates.
Capacity is N records per rank for all policies to avoid policy-specific
pool overflow. Report this conservative memory provisioning.

Measured-cost selector: choose a policy per pattern/count using calibration
medians only, apply unchanged to paired test seed. Compare a single globally
best calibration policy and each fixed candidate. Charge selected map cost,
setup and per-run selection; show calibration amortization separately.

Prior: fixed ridge regression (alpha=10 after feature standardization) of
log regional compute, boundary bytes, and window runtime using topology,
route plans, and departure schedules only. Fit on calibration cases; test on
held-out seed. No hyperparameter search on test data. Predict all candidate
windows and choose minimum sum plus candidate-generation cost. Charge ALL
candidate maps and features for this predictor, not only the winning map.
Assess persistence by consecutive-window cost changes and winning-policy
streaks; traffic cohorts never split across train and test.

Gate: >=5% lower complete cost than the best fixed calibration policy,
with a positive lower 95% case-bootstrap bound on held-out mean relative gain.
Report oracle hindsight as an unattainable upper reference, not as prediction.
No causal claim that speed/density alone explains GPU cost. No new Nsight
hardware-counter claims: CUDA-event diagnostics and wall time are distinct.

Budget: cumulative previous posted spend $7.90028261. New hard reservation
$6.40; cumulative ceiling $15. One container, three A100s, no retries,
function timeout 1900 s plus 120 s startup/60 s cleanup at $0.003/s + $0.16.
