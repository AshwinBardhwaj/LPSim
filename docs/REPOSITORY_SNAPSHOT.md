# Private development snapshot

This repository contains the current Berkeley/Jetson LPSim source, network inputs,
tests, visualization tools, and offline report generators. It was prepared from
https://github.com/Xuan-1998/LPSim at upstream commit
`2fc388a4f638144e92f870f1acfa9f6e48c57ea5`, with local development changes.
The upstream license and attribution are retained. The new repository begins
with a source snapshot; it does not copy the upstream Git object history.

Build outputs, installed environments/packages, caches, generated routes,
trajectories, playback data, profiler captures, and reports are excluded. Network
CSV inputs are included. `lpsim_env` is project source, not a virtual environment;
the C++ headers under `src/routing/sp` are build dependencies and are retained.

See the main README for build/network generation, `viz/TRAJECTORIES.md` for
trajectory processing, and `docs/RUN_REPORTS.md` for report generation. Generated
visualizations need their data rebuilt locally before they can display a run.
