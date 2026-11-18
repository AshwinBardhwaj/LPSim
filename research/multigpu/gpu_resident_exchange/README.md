# GPU-resident exchange experiment

Experimental implementation of persistent local vehicle/ghost slots and CUDA
GPU-side peer-message consumption. Read [AUDIT.md](AUDIT.md) for exact scope,
dependency reasoning and remaining limitations. **No physical-GPU correctness or
performance result has been established for this backend.**
See [STATUS.md](STATUS.md) for completed local logical-partition validation.

Files:

- `prepare.py`: freezes the existing physics closure and both baseline sources;
  generates minimal global-ID adapters with checked source substitutions; hashes
  every frozen source file.
- `resident.cu`, `slots.cuh`, `protocol.h`: persistent slots, device counts,
  epoch-tagged messages, neighbor event dependencies, explicit buffer consumption
  and local-lane spawn scheduling. Errors propagate without indefinite polling.
- `protocol_tests.cu`: GPU tests for empty messages, refresh without allocation,
  removal, reuse, duplicates, overflow, stale epochs and wrong-phase ownership.
- `evaluate.py`: exact-gated comparison with both single-GPU baselines, mode 3,
  rebuild/barrier, slots/barrier and slots/neighbor. Does not launch cloud resources.
- `analyze.py`: paired-repeat bootstrap comparisons against both single-GPU
  baselines and mode 3; rejects logical or incomplete runs.
- `test_tools.py`: CPU checks of source freezing and evaluation safety gates.

The transport is **CUDA P2P, not NVSHMEM**. It is compileable with the available
CUDA toolkit. NVSHMEM remains a subsequent transport integration requiring its SDK
and physical device validation. The implementation intentionally does not claim
that compiling code establishes exact simulation equivalence.

## Build and check

Run from the repository root. Each output directory must be new:

```bash
python3 -m unittest discover -s research/multigpu/gpu_resident_exchange -p 'test_*.py'
bash research/multigpu/gpu_resident_exchange/build.sh /tmp/resident-build sm_80
/tmp/resident-build/protocol-tests
```

Use `sm_87` for Jetson. Logical ranks on a single GPU check arithmetic and
membership only; they do not validate physical peer access or scaling.

```bash
python3 research/multigpu/gpu_resident_exchange/evaluate.py \
  --build /tmp/resident-build \
  --model artifacts/exchange-study-20261029/inputs/models/berkeley_junction_study/model.txt \
  --snapshot artifacts/exchange-study-20261029/inputs/berkeley_8192/snapshot.bin \
  --owners artifacts/exchange-study-20261029/inputs/berkeley_8192/paper_balanced_3.txt \
  --ranks 3 --steps 60 --verify-only --output /tmp/resident-verification
```

Add `--logical` only on a single CUDA device. On physical GPUs, removing
`--verify-only` runs excluded warmups and five shuffled timing repetitions after
every variant passes exact verification and final hashes match. Each subprocess
has a timeout; partial receipts/logs survive failure. Compare the SF and developed
trajectory cases as well before drawing conclusions. Historical cloud budgets
are not reused by this runner.

Resident CLI:

```text
resident MODEL SNAPSHOT OWNERS STEPS NEW_OUT RANKS VERIFY barrier|neighbor CAPACITY
```

`CAPACITY=0` selects bounded initial capacity; nonzero sets per-rank Car slots.
`LPSIM_SLOT_REBUILD=1` enables the full membership sort/gather ablation.
Input schema/ABI are inherited from the frozen research simulator; this is not a
general-purpose untrusted snapshot loader.

## Measurement limits

Timings include final device completion, exclude setup and final verification/
serialization, and report whole-window wall time. Verified runs include oracle
work and must not be used as performance samples. The existing baseline reports
its sum of per-step timed windows; the resident backend reports a whole window
without intermediate output. Preserve this timing-scope distinction in reports.

The runner reports raw paired repeats and medians. Run `analyze.py RUN_DIR` for
paired-repeat intervals; it does not automatically promote a backend.
Detailed component traces, hardware telemetry,
NVSHMEM comparison and broader overlap are outstanding. Zero normal-path count
D2H operations is a structural code property, not a measured profiling counter.
