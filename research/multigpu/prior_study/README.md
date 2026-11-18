# Longer demand and pre-run partition selection

Read PROTOCOL.md for the frozen experimental design and gate. This study uses
three physical GPUs and static ownership maps; it does not implement dynamic
repartitioning. No new GPU hardware-counter claims are made.

1. `python prepare.py` freezes demand inputs.
2. `maps.py INPUTS OUTPUT` generates the same three candidates and topology /
   departure-only features. Dependencies: numpy 1.26.4, pymetis 2023.1.1.
3. `launch.py` makes one budget-guarded Modal call. **Do not rerun casually:**
   update the cumulative billing ledger before another authorized launch.
4. Extract the returned archive and verify its SHA-256 manifest.
5. `python analyze.py /absolute/path/to/extracted/run` evaluates calibration
   selection and a fixed ridge model on the held-out seed.
6. `python supplement.py` regenerates the explicitly post-hoc same-map diagnostic.
7. `MPLCONFIGDIR=/tmp/lpsim-mpl python report.py` creates the figures/PDF.

Physics comes from the frozen multigpu-preparation source. Changes from
exchange_study/local.cu are full-N pool capacity (all policies), 30 s population
sampling/physical checks, sampled diagnostic events, and cumulative wire logs.
Every standard step still executes the same kernels and state exchanges.

Use protocol source/input hashes and the returned image, hardware and binary
records for reproduction. No test features include measured traffic state.
