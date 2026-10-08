# Research workspace and milestone retention

The main Git checkout contains source, tests, versioned network definitions and
small milestone records. Large recorded evidence lives in the sibling
`../research-archive/` directory. `artifacts/current/` is writable scratch space;
old `artifacts/<milestone>` and `reports/<report>` paths are compatibility links.
Do not run old output-generating commands against those frozen paths.

## Start a new experiment

Commit the intended code, then run:

```
python3 tools/research.py list
python3 tools/research.py new-run signal-ablation
```

The command creates a writable source snapshot in
`artifacts/current/<timestamp>-signal-ablation/workspace/`, records the exact
commit and does not launch a simulation. Build and run inside that workspace.
The existing dated experiment scripts then write into that isolated workspace,
not into historical evidence. Copy required cached inputs from the archive when
reconstructing a raw-OSM experiment; cached small GraphML inputs are versioned.

A milestone is promoted only after validation. It must receive a new unique ID,
exact source/input hashes, commands, seeds, runtime/tool versions and power
settings, raw metrics, validation and diagnostic output, profiling captures,
reports and a working playback bundle. Never overwrite or relabel an earlier
result as though it came from a later code version.

## Storage and integrity

`research-archive/catalog.json` and the milestone index describe preserved work.
`milestones/` contains conventional file trees. Identical regular files share
read-only objects from `objects/`, with executable status included in the object
identity. This is exact-content deduplication, not scientific-data deletion.

```
python3 tools/research.py verify
python3 tools/research.py verify parallel-junctions
```

Hash validation detects later modification even if someone manually changes
permissions. Read-only permissions are workflow protection, not a WORM storage
or cryptographic immutability guarantee. Runtime logs and server PID files live
under `research-archive/runtime/`, outside milestone content.

`backups/20261008/` contains a separately stored, verified compressed pre-cleanup
workspace copy and Git recovery bundle. These copies are on the SAME physical
device: they protect against cleanup mistakes, not device failure. Keep them
until an off-device copy has been verified. Do not prune Git history or delete
these backups as routine cleanup.

## What may be removed

Build products, Python caches and abandoned temporary runs may be removed after
checking that they contain no unique diagnostics. Preserve failed scenarios that
demonstrate fixed defects and all repetitions supporting published statistics.
Source changes must be committed or checkpointed before resetting a checkout.
An older milestone may be superseded but should remain available and explicitly
labeled. Unknown original seeds/builds remain documented provenance gaps.

The older Berkeley recording has incomplete original-run provenance. Preserved
recordings/reports can be inspected and replayed visually, but they are not
presented as a fully reconstructible original simulation.

## Replay a preserved San Pablo milestone

```
python3 tools/research.py replay parallel-junctions --case network_20
```

This uses the retained milestone executable and frozen inputs, writes fresh output
under `artifacts/current/`, and compares behavioral metrics and recorded CSVs.
It requires a compatible Jetson/CUDA environment. Timings are deliberately not
required to match. All five San Pablo 20-trip preservation replays matched their
recorded CSVs exactly during cleanup; this does not validate every dense case.

## View reports and restart visualization

The milestone index is at `http://100.85.134.52:8093/`. Existing playback URLs on
ports 8090, 8091 and 8092 retain their original `/web/` paths. Start servers with:

```
python3 tools/serve_research.py --bind 100.85.134.52
```

This binds only the Jetson's private Tailscale address. Without `--bind`, servers
use localhost. These are background processes, not installed reboot services.
The index exposes milestone packages, not the backup directory. Berkeley's
historical web frontend may still require its original external JavaScript CDNs.

## Cleanup checkpoint, 2026-10-08

Six milestone packages preserve 2,605 inventoried files (6.83 GB logical), using
5.75 GB of unique file content after exact deduplication. A verified 2,863-file
pre-cleanup backup and Git bundle precede the migration. The temporary push
worktree and abandoned Git packfiles receive separate verified backup archives.
The source preservation and reconciliation commits are `ae66447` and `a759195`.
The versioned `research/milestones/` registry records validation and provenance.
Canonical network inputs remain under `data/networks/`.
