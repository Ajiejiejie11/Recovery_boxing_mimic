#!/bin/bash
set -ex

# undo separate folder for tzcode (see recipe.yaml), which is necessary because rattler-build
# won't pull both sources into the same directory due to overlapping metadata-files
mv tzcode/* .

# by default the makefile does not install leap-seconds.list;
# however, some implementations (e.g. libc++) rely on it so
# we expand the default (TZDATA_TEXT=tzdata.zi leapseconds)

make -e \
  DESTDIR=./build \
  EXPIRES_LINE=1 \
  USRDIR='' \
  TZDATA_TEXT='tzdata.zi leapseconds leap-seconds.list' \
  install

mkdir -p "${PREFIX}/share"
mv ./build/share/zoneinfo "${PREFIX}/share/"
