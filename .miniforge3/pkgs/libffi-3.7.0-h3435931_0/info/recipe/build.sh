#!/usr/bin/env bash

set -e -x
shopt -s extglob

for file in $(find ${SRC_DIR} -type f -name "config.sub"); do
  cp $BUILD_PREFIX/share/gnuconfig/config.sub ${file}
done
for file in $(find ${BUILD_PREFIX}/Library/usr/share -type f -name "config.sub"); do
  cp $BUILD_PREFIX/share/gnuconfig/config.sub ${file}
done

export CFLAGS="${CFLAGS//-fvisibility=+([! ])/}"
export CXXFLAGS="${CXXFLAGS//-fvisibility=+([! ])/}"

configure_args=(
    --disable-debug
    --disable-dependency-tracking
    --prefix="${PREFIX}"
    --includedir="${PREFIX}/include"
    --disable-exec-static-tramp
)

if [[ "$target_platform" != win-* ]]; then
  configure_args+=(--build=$BUILD --host=$HOST)
else
  configure_args+=(--disable-static)
  export CPPFLAGS="$CPPFLAGS -DFFI_BUILDING_DLL"
fi

autoreconf -vfi

if [[ "$target_platform" == linux* ]]; then
  # this changes the install dir from ${PREFIX}/lib64 to ${PREFIX}/lib
  sed -i 's:@toolexeclibdir@:$(libdir):g' Makefile.in */Makefile.in
  sed -i 's:@toolexeclibdir@:${libdir}:g' libffi.pc.in
fi

if [[ "$target_platform" == win-64 ]]; then
  export host_alias=x86_64-pc-windows-msvc
  export CFLAGS="$CFLAGS --target=x86_64-pc-windows-msvc"
elif [[ "$target_platform" == win-arm64 ]]; then
  export host_alias=aarch64-pc-windows-msvc
  export CFLAGS="$CFLAGS --target=aarch64-pc-windows-msvc"
fi

./configure "${configure_args[@]}" || { cat ${host_alias}/config.log; exit 1;}
if [[ "$target_platform" == win-* ]]; then
  pushd ${host_alias}
    patch_libtool
    sed -i.bak 's/|-fuse-ld/|-Xclang|-fuse-ld/g' libtool
  popd
fi

make -j${CPU_COUNT} ${VERBOSE_AT}
make check
make install

# This overlaps with libgcc-ng:
rm -rf ${PREFIX}/share/info/dir

if [[ "$target_platform" == win-* ]]; then
  mv $PREFIX/lib/ffi.dll.lib $PREFIX/lib/libffi.dll.lib
fi
