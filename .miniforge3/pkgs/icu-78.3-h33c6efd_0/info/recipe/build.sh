#!/bin/bash
# Get an updated config.sub and config.guess
cp $BUILD_PREFIX/share/libtool/build-aux/config.* ./source

set -ex

which link

if [[ "${target_platform}" == win-* ]]; then
  # Ensure that MSVC come before MSYS2
  export PATH="$(dirname "$(which -a link | grep MSVC | head -1)"):$PATH"
  export CXXFLAGS="$CXXFLAGS /std:c++17"
  export CXXFLAGS_FOR_BUILD="$CXXFLAGS_FOR_BUILD /std:c++17"
  # Tell it we're building for MSVC
  cp source/config/mh-msys-msvc source/config/mh-unknown
  # EXEXT is wrong for *-pc--windows
  sed -i.bak 's/EXEEXT=""/EXEEXT=.exe/g'  source/configure
  if [[ "${build_platform}" == "win-64" ]]; then
    export BUILD=x86_64-pc-windows
  elif [[ "${build_platform}" == "win-arm64" ]]; then
    export BUILD=aarch64-pc-windows
  fi
  if [[ "${target_platform}" == "win-64" ]]; then
    export HOST=x86_64-pc-windows
  elif [[ "${target_platform}" == "win-arm64" ]]; then
    export HOST=aarch64-pc-windows
  fi
fi

echo ${target_platform} ${build_platform}

cd source

chmod +x configure install-sh

EXTRA_OPTS="${EXTRA_OPTS:-}"

if [[ "$CONDA_BUILD_CROSS_COMPILATION" == "1" ]]; then
    mkdir cross_build
    pushd cross_build
    (
    export LIB=$LIB_FOR_BUILD
    export INCLUDE=$INCLUDE_FOR_BUILD
    CC=$CC_FOR_BUILD CXX=$CXX_FOR_BUILD AR= AS= LD= CFLAGS= CXXFLAGS=$CXXFLAGS_FOR_BUILD LDFLAGS=$LDFLAGS_FOR_BUILD CPPFLAGS= ../configure \
      --build=${BUILD} \
      --host=${BUILD} \
      --disable-samples \
      --disable-extras \
      --disable-layout \
      --disable-tests || (cat config.log; exit 1)
    make -j${CPU_COUNT}
    )
    EXTRA_OPTS="$EXTRA_OPTS --with-cross-build=$PWD"
    popd
fi

if [[ ${HOST} =~ .*darwin.* ]]; then
  EXTRA_OPTS="$EXTRA_OPTS --enable-rpath"
fi

./configure --prefix="${PREFIX}"  \
            --build=${BUILD}      \
            --host=${HOST}        \
            --disable-samples     \
            --disable-extras      \
            --disable-layout      \
            --disable-tests       \
            ${EXTRA_OPTS} || (cat config.log; exit 1)

make -j${CPU_COUNT} ${VERBOSE_CM}
if [[ "${CONDA_BUILD_CROSS_COMPILATION}" != "1" ]]; then
  make check
fi
make install

rm -rf ${PREFIX}/sbin
