#!/usr/bin/env bash
set -euo pipefail
study_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${1:?Usage: build.sh NEW_OUTPUT_DIR sm_80_or_sm_87}"
arch="${2:?Specify CUDA architecture}"
case "$arch" in sm_80|sm_87) ;; *) echo "Unsupported architecture" >&2; exit 2;; esac
python3 "$study_dir/prepare.py" "$build_dir"
nvcc_bin="${NVCC:-/usr/local/cuda/bin/nvcc}"
flags=(-O3 --use_fast_math -lineinfo -std=c++17 "-arch=$arch" -Xcompiler -pthread)
"$nvcc_bin" "${flags[@]}" "$build_dir/study/resident.cu" -o "$build_dir/resident"
"$nvcc_bin" "${flags[@]}" "$build_dir/study/protocol_tests.cu" -o "$build_dir/protocol-tests"
"$nvcc_bin" "${flags[@]}" "$build_dir/baseline/local.cu" -o "$build_dir/baseline-local"
"$nvcc_bin" "${flags[@]}" "$build_dir/baseline/original.cu" -o "$build_dir/baseline-original"
"$nvcc_bin" --version > "$build_dir/nvcc-version.txt"
printf '%s\n' "${flags[@]}" > "$build_dir/build-flags.txt"
sha256sum "$build_dir/resident" "$build_dir/protocol-tests" "$build_dir/baseline-local" "$build_dir/baseline-original" > "$build_dir/binaries.sha256"
