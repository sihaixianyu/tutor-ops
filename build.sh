#!/bin/sh

set -eu

if [ "${1:-}" = "-r" ]; then
    rm -rf build
    shift
fi

cmake --preset dev
cmake --build --preset dev "$@"
