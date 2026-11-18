# Physical A100 partition baseline study

This experiment uses the frozen `artifacts/multigpu-preparation-20261027` source/input bundle, not unrelated current application changes. Read `PROTOCOL.md` before interpreting results. It adds concurrent direct-peer transport while retaining the original physics and dense stable-ID layout.

- `concurrent.cu`: includes the frozen correctness implementation; adds persistent CPU launch workers, destination-packed state payloads, ownership delta broadcasts and peer reservation exchange. `single` uses the same motion/index/junction calculations without peer barriers. `verify=1` runs an independent exact reference and gap/conflict/ghost checks. `instrument=1` records populations and CUDA-event stage durations outside unprofiled timing.
- `modal_study.py`: pinned CUDA image, 3 A100s, one container, no retries, bounded timeouts, automatic raw archive and hardware/telemetry recording. Explicit A100-80GB was selected after a 40GB request remained unallocated.
- `launch.py`: client watchdog and conservative budget reservation. Existing results remain intact; every output directory is new.
- `run_matrix.py`: physical gate, then frozen training cases. Primary performance window is 180–600 s after a common 0–180 s prefix. Assignments use only the observed 180 s population. One warm-up plus five repeats; final physical hashes must match across policies.
- `profile.py`: separate Nsight Compute kernel replay on saved states, `--clock-control none`, with pre-step populations and raw counter reports. It measures motion, junction arbitration and leader-key/link kernels; it does not replace a whole-program timeline.
- `collect.py`: archive path checks and SHA-256 verification.
- `analyze.py`, `report.py`, `presentation.py`: local CSVs, figures, PDF and presentation script.

Reproduce cloud phases using the launcher, subject to a newly authorized budget:

```
python3 research/multigpu/a100_study/launch.py artifacts/NEW/gate gate
python3 research/multigpu/a100_study/launch.py artifacts/NEW/concurrent-gate concurrent_gate
python3 research/multigpu/a100_study/launch.py artifacts/NEW/matrix8192 matrix_8192
```

The profiling script locates saved matrix output in the Modal artifact volume; inspect its case selection before reuse. Local analysis currently targets the dated study directories intentionally, to avoid mixing milestones.

Memory limitations: N vehicle slots and immutable network/routes are replicated per device. Peer state pool capacity is fixed at 2N per destination. Message counts are copied to pinned host memory; compact ownership deltas broadcast to all ranks. Conservative phase barriers remain. Bulk radix indexing is identical for all policies; the separately selectable incremental reference is not combined with these comparisons.

Safety/correctness limitations: this is research code with fixed valid frozen inputs, not a general-purpose public snapshot loader. A passing development matrix is not held-out validation of a learned policy, and an arithmetic balance screen is not an implemented oracle schedule.

Final local deliverables are under `reports/a100-study-20261028/`: the report PDF, eight standalone figures, all derived CSV tables, editable/printable presentation notes, and presentation/reproducibility ZIPs. The full ZIP preserves repository-relative paths; extract it into an empty directory, then run `collect.py`, `analyze.py`, `report.py`, and `presentation.py` from its root. Python dependencies are NumPy, pandas and matplotlib. Raw archives contain Nsight `.ncu-rep` files and exported counter CSVs; NVIDIA Nsight Compute is needed only to inspect/re-export the native reports. No cloud compute is needed to read the PDFs or rerun local analysis. `bundle-manifest.json` hashes each packaged source/input/evidence file.

The final small-population allocation records NVLink link status, but container topology enumeration remains unavailable. Link status is not a measurement of application bandwidth or proof of exact NVSwitch wiring. See the report's interconnect limitations.
