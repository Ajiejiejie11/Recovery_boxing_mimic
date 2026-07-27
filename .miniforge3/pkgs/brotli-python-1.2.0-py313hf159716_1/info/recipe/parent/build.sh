#!/bin/bash

set -exuo pipefail

BROTLI_CFLAGS="-O3"
if [[ "${QEMU_LD_PREFIX:-}" != "" ]]; then
  EXTRA_CMAKE_FLAGS=-DBROTLI_WRAPPER_LD_PREFIX="${QEMU_LD_PREFIX:-}"
fi

# Build both static and shared libraries
cmake ${CMAKE_ARGS} -DCMAKE_INSTALL_PREFIX=$PREFIX \
      -DCMAKE_C_FLAGS=$BROTLI_CFLAGS \
      -GNinja \
      ${EXTRA_CMAKE_FLAGS:-} \
      .

ninja
if [[ "${CONDA_BUILD_CROSS_COMPILATION:-}" != "1" || "${CROSSCOMPILING_EMULATOR:-}" != "" ]]; then
  ctest -V
fi
