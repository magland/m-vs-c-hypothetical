#!/usr/bin/env bash
# Compile the C benchmark.  Tweak CFLAGS via the environment if you like.
set -euo pipefail
cd "$(dirname "$0")"

CC="${CC:-gcc}"
CFLAGS_COMMON="${CFLAGS_COMMON:--O3 -march=native -funroll-loops -fopenmp}"

echo "+ ${CC} ${CFLAGS_COMMON} -o lap2d_green lap2d_green.c -lm"
${CC} ${CFLAGS_COMMON} -o lap2d_green lap2d_green.c -lm

echo "+ ${CC} ${CFLAGS_COMMON} -ffast-math -o lap2d_green_fast lap2d_green.c -lm"
${CC} ${CFLAGS_COMMON} -ffast-math -o lap2d_green_fast lap2d_green.c -lm
