#!/bin/bash

set -exo pipefail

export CFLAGS="${CFLAGS} -O3 -fPIC"

make -j$CPU_COUNT -C contrib/pzstd all

declare -a _CMAKE_EXTRA_CONFIG

if [[ ${HOST} =~ .*linux.* ]]; then
  # I hate you so much CMake.
  LIBPTHREAD=$(find ${PREFIX} -name "libpthread.so")
  _CMAKE_EXTRA_CONFIG+=(-DPTHREAD_LIBRARY=${LIBPTHREAD})
  LIBRT=$(find ${PREFIX} -name "librt.so")
  _CMAKE_EXTRA_CONFIG+=(-DRT_LIBRARIES=${LIBRT})
fi

if [[ "$PKG_NAME" == *static ]]; then
  ZSTD_BUILD_STATIC=ON
  # cannot build CLI without shared lib
  ZSTD_BUILD_SHARED=ON
  ZSTD_ZLIB_SUPPORT=OFF
else
  ZSTD_BUILD_STATIC=OFF
  ZSTD_BUILD_SHARED=ON
  # For .gz support
  ZSTD_ZLIB_SUPPORT=ON
fi

pushd build/cmake
  FULL_AR=`which ${AR}`
  cmake -GNinja                            \
        -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
        -DCMAKE_INSTALL_LIBDIR="lib"       \
        -DCMAKE_PREFIX_PATH="${PREFIX}"    \
        -DCMAKE_AR=${FULL_AR}              \
        -DZSTD_BUILD_STATIC=${ZSTD_BUILD_STATIC} \
        -DZSTD_BUILD_SHARED=${ZSTD_BUILD_SHARED} \
        -DZSTD_PROGRAMS_LINK_SHARED=ON     \
        -DZSTD_ZLIB_SUPPORT=${ZSTD_ZLIB_SUPPORT} \
        "${_CMAKE_EXTRA_CONFIG[@]}"

  ninja install
popd

# Just remove all installed documentation since users of conda-forge haven't
# asked for it.
rm -rf ${PREFIX}/share/man/man1/unzstd*
rm -rf ${PREFIX}/share/man/man1/zstd*
