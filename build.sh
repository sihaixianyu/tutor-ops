#!/bin/sh

set -eu

if [ "${1:-}" = "-r" ]; then
    rm -rf build/CMakeCache.txt build/CMakeFiles
    shift
fi

cmake --preset dev
cmake --build --preset dev "$@"
