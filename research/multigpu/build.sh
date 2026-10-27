#!/usr/bin/env bash
set -euo pipefail
src_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
build_dir="${1:?Usage: build.sh OUTPUT_DIR sm_87_or_sm_80}"
cuda_arch="${2:?Specify sm_87 for Jetson or sm_80 for A100}"
case "$cuda_arch" in sm_87|sm_80) ;; *) echo "Unsupported architecture: $cuda_arch" >&2; exit 2;; esac
mkdir -p "$build_dir"
nvcc_bin="${NVCC:-/usr/local/cuda/bin/nvcc}"
flags=(-O3 --use_fast_math -lineinfo -std=c++17 "-arch=$cuda_arch")
"$nvcc_bin" "${flags[@]}" "$src_root/research/multigpu/partitioned.cu" -o "$build_dir/partitioned"
"$nvcc_bin" "${flags[@]}" "$src_root/src/simulator/scalable_junction_simulator.cu" -o "$build_dir/reference"
"$nvcc_bin" "${flags[@]}" "$src_root/tools/incremental_simulator.cu" -o "$build_dir/reference_incremental"
"$nvcc_bin" "${flags[@]}" "$src_root/research/multigpu/hardware_probe.cu" -o "$build_dir/hardware_probe"
"$nvcc_bin" --version > "$build_dir/nvcc-version.txt"
printf '%s\n' "${flags[@]}" > "$build_dir/build-flags.txt"
sha256sum "$build_dir/partitioned" "$build_dir/reference" "$build_dir/reference_incremental" "$build_dir/hardware_probe" > "$build_dir/binaries.sha256"
