# Local multi-GPU processing and paper-style baselines

This experiment is separate from `a100_study`; the old source/input freeze is preserved.

- `policies.py`: one partition interface for geography, traffic-weighted METIS and Leiden plus geographic k-means, with explicit seeds and construction metadata. No dependency fallback. Paper-style, not an exact reproduction of unspecified original settings.
- `prepare_network.py`: converts cached SF-partial CSVs into the same research junction representation. Writes real derived network files under `data/networks/sf_partial_local_scaling`. It does not overwrite an existing model. The original CSV hashes and lane-count limitation are documented.
- `prepare_cases.py`: constructs safe active populations and freezes policy maps under `artifacts/local-scaling-20261028/inputs`. These are performance initial conditions, not calibrated traffic demand.
- `local.cu`: includes the frozen simulator, replaces dense vehicle processing with sorted local owned/ghost records, and retains exact simulation calculations. Persistent capacities depend on initial local population; all overflow paths abort. It does not yet grow pools online. Network-index/summaries remain replicated. Full-record exchange and compact-list rebuilding costs are deliberately visible.
- `run.py`: physical-device correctness gate, original single-GPU reference, local 1/2/3-GPU comparisons, excluded warm-up and three repetitions. Raw output hashes must match.
- `modal_study.py`, `launch.py`: bounded single-use 3-A100 container and cumulative budget guard. One- and two-device tests use a subset of that allocation to keep hardware conditions matched; billing still charges three reserved GPUs.
- `analyze.py`: verified archive extraction, timing/population tables, scaling figures and PDF. No GPU allocation is needed for local analysis.

See `PROTOCOL.md` for scope and interpretation. The primary scaling pilot is 60 timesteps, not a complete 600-second benchmark. Population weak scaling uses fixed SF topology. Do not describe it as full topology weak scaling or production-ready distributed storage.

Local build uses the layout `/tmp/lpsim-local-scaling/source` (frozen source tree) and `/tmp/lpsim-local-scaling/study/local.cu`, with `nvcc -O3 --use_fast_math -lineinfo -std=c++17 -arch=sm_87 -Xcompiler -pthread`. Cloud build uses the same flags with `sm_80`.

Partition dependencies: PyMetis 2023.1.1, igraph 0.11.8, leidenalg 0.10.2; NumPy; network generation additionally uses NetworkX and Shapely. Full local environment capture is preserved. Stable global priority is preserved by sorting local slots in ascending global ID; local indices are never treated as persistent identities across exchanges.
