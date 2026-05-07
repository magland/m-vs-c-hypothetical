#!/usr/bin/env bash
# Run the C benchmark and the MATLAB benchmark, printing each side's
# fingerprint so they can be compared by eye (or with `diff`).
set -euo pipefail
cd "$(dirname "$0")"

bash ./build.sh

echo
echo "============== C  strict     (1 thread)  =============="
OMP_NUM_THREADS=1 ./lap2d_green

echo
echo "============== C  strict     (8 threads) =============="
OMP_NUM_THREADS=8 ./lap2d_green

echo
echo "============== C  fast-math  (1 thread)  =============="
OMP_NUM_THREADS=1 ./lap2d_green_fast

echo
echo "============== C  fast-math  (8 threads) =============="
OMP_NUM_THREADS=8 ./lap2d_green_fast

echo
echo "================== MATLAB =================="
echo "(first call = warm-up; subsequent calls are the timings of record)"
matlab -batch "lap2d_green(1); fprintf('\n--- WARM-UP DONE ---\n\n'); lap2d_green(1); fprintf('\n--------\n\n'); lap2d_green(8)"
