# Multi-GPU preparation: a validated three-partition protocol

This directory prepares LPSim for testing on three A100 GPUs. It implements independent partition buffers and explicit host-staged boundary messages on the Jetson, with the original simulator running separately as a timestep-by-timestep oracle. It does not yet implement an optimized concurrent peer-transport backend or demonstrate a multi-GPU speedup.

Read [protocol.md](protocol.md) for ownership, dependency stages and wire formats; [evaluation.md](evaluation.md) for the frozen experiment rules. Results and the portable source/input snapshot are in `reports/multigpu-preparation-20261027` and `artifacts/multigpu-preparation-20261027` respectively.

## Implemented components

| Item | Implementation |
|---|---|
| Frozen single-GPU reference | Original compact-state/radix/indexed-following backend, source closure + input/output SHA-256, reproducible build script |
| Independent incremental option | `reference_incremental` binary with `LPSIM_INDEX_POLICY=incremental`, and `partitioned ... incremental` |
| Ownership and dependencies | Whole-junction ownership; incoming roads local; receiving-road ghosts; authoritative reservation exchange before spawn |
| Three logical partitions | Separate current/next state and indexes; explicit state, ownership and reservation exchange; no reference-fed physics |
| Policies | Geography, observed active-vehicle count, balanced communication-weighted graph heuristic |
| Pools | Persistent device and pinned host state-message buffers; capacity checks; deterministic transfer routing |
| Workloads | 27 fresh train/validation/test cases plus 3 historical correctness snapshots, all hashed |
| Cloud preparation | Digest-pinned CUDA image, SDK pin, sm_80 build, hardware/peer-copy probe, counter-permission probe, correctness and artifact runner |

Dense stable-ID slots and replicated immutable network/routes are intentional prototype limitations. Memory distribution and concurrency must be measured after sparse local storage and direct peer transport are added. Full ownership-map copies and reference verification are accounted separately; none of the Jetson prototype timings is a scaling benchmark.

## Local reproduction

From the repository root, the existing frozen bundle can be verified without changing it:

```bash
python3 research/multigpu/verify_bundle.py artifacts/multigpu-preparation-20261027
python3 research/multigpu/test_preparation.py
```

To rebuild from the frozen source into a new directory:

```bash
bash artifacts/multigpu-preparation-20261027/source/research/multigpu/build.sh \
  /tmp/lpsim-multigpu-rebuild sm_87
```

`sm_87` is for the Jetson, `sm_80` for the A100. The toolchain flags are `-O3 --use_fast_math -lineinfo -std=c++17`. CUDA, compiler, device and current 15 W power-mode records are included in the bundle. Validation runs did not change power or clock settings and are not controlled performance repetitions. Any later benchmark must explicitly set and record its chosen clocks/power policy.

A minimal populated-state check (create a fresh output directory):

```bash
mkdir -p /tmp/lpsim-partition-check
/tmp/lpsim-multigpu-rebuild/partitioned \
  artifacts/multigpu-preparation-20261027/inputs/network/model.txt \
  artifacts/multigpu-preparation-20261027/inputs/reference_8192_t180/snapshot.bin \
  artifacts/multigpu-preparation-20261027/inputs/reference_8192_t180/geography.txt \
  120 /tmp/lpsim-partition-check bulk 0 0,0,0
```

The CLI is `MODEL SNAPSHOT OWNERS STEPS OUTPUT_DIR bulk|incremental [controller_mode] [devices] [message_capacity]`. Timestep is fixed at 0.25 s. Device IDs `0,1,2` enable the same correctness protocol across three physical GPUs. Physical partitions still run sequentially and use host staging in this version.

To create an independent fresh preparation bundle and run the complete suite:

```bash
python3 research/multigpu/prepare.py artifacts/multigpu-reproduction
bash research/multigpu/build.sh artifacts/multigpu-reproduction/bin sm_87
python3 research/multigpu/validate.py artifacts/multigpu-reproduction \
  artifacts/multigpu-reproduction/bin/partitioned
```

Preparation currently needs the existing Berkeley model/route pool and historical compact-motion snapshots. A distributed frozen bundle already contains all inputs and does not need those old artifact paths to run/build/verify.

## Modal preflight when GPU access is available

The code has been imported successfully using Modal SDK 1.6.3. No Modal image has been built remotely and no A100 function has been launched during this preparation. The CUDA Linux/amd64 image is pinned by digest in `container-lock.json`; Jetson ARM binaries are rebuilt for A100.

```bash
python3 -m venv /tmp/lpsim-modal
/tmp/lpsim-modal/bin/pip install 'modal==1.6.3'
# Authenticate using your Modal account when access is available.
export LPSIM_BUNDLE="$PWD/artifacts/multigpu-preparation-20261027"
/tmp/lpsim-modal/bin/modal run research/multigpu/modal_runner.py \
  --case reference_8192_t180 --policy geography --steps 120
```

The last command requests **three A100-40GB GPUs and incurs cloud charges**. It is supplied for the access stage, not run by preparation. The runner verifies hashes; records `nvidia-smi`, topology and directional peer-copy tests; probes Nsight permissions; compares logical and physical partition outputs; and commits logs, summaries, traces if available, hashes and a tarball to the `lpsim-multigpu-artifacts` Modal Volume, including failures. It prints the archive name for retrieval:

```bash
/tmp/lpsim-modal/bin/modal volume get lpsim-multigpu-artifacts RUN_ID.tar.gz ./
```

SDK dependency versions from the local import test are recorded in `modal-requirements-jetson.lock`; that is an ARM/Python-3.10 environment record, not a claim of an identical x86/Python-3.11 environment. The container pins CUDA and uses Modal's Python 3.11 image setup. Record the actual Modal image build ID and runtime packages at the first cloud build.

Official API references used: [GPU counts and A100 variants](https://modal.com/docs/guide/gpu), [digest-compatible existing images and Python setup](https://modal.com/docs/guide/existing-images), [local source copies](https://modal.com/docs/guide/images), [persistent Volumes](https://modal.com/docs/guide/volumes). A three-GPU request is for devices on one physical host; topology and peer access are still measured rather than assumed.

## First work after access

1. Pass the prepared hardware and physical-device correctness gate.
2. Add concurrent per-device execution and direct peer transport using the tested protocol; keep a measured host fallback.
3. Add a timing-only path, separate correctness repetitions, and full stage/communication accounting. Use the strongest fixed policy as the baseline.
4. Measure a cost-informed oracle before training a prior. Apply the preregistered held-out 10% complete-runtime improvement gate.
