#!/bin/bash

set -ex

if [[ "$target_platform" == "win-"* ]]; then
  IS_WIN=1
  LIBRARY_PREFIX="/Library"
  EXE_EXT=".exe"
else
  IS_WIN=0
  LIBRARY_PREFIX=""
  EXE_EXT=""
fi

TOOLS="addr2line ar c++filt elfedit nm objcopy objdump ranlib readelf size strings strip"
if [[ "${cross_target_platform}" != "osx-"* ]]; then
  TOOLS="${TOOLS} as gprof ld"
fi
if [[ "${cross_target_plaform}" == "win-"* ]]; then
  TOOLS="${TOOLS} dlltool dllwrap windmc windres"
fi

CHOST="${triplet}"

TOOLS_ACTIVATE=""
TOOLS_DEACTIVATE=""

for tool in ${TOOLS}; do
  tool_upper=$(echo "${tool}" | tr 'a-z+-.' 'A-ZX__')
  TOOLS_ACTIVATE="${TOOLS_ACTIVATE} \\\\\n  \"${tool_upper},${CHOST}-${tool}\""
  TOOLS_DEACTIVATE="${TOOLS_DEACTIVATE} \\\\\n  \"${tool_upper}\""
done
cp ${RECIPE_DIR}/scripts/* .

find . -name "*activate*.*" -not -name "*.bak" -exec sed -i.bak "s|@IS_WIN@|${IS_WIN}|g"                   "{}" \;
find . -name "*activate*.*" -not -name "*.bak" -exec sed -i.bak "s|@TOOLS_ACTIVATE@|${TOOLS_ACTIVATE}|g"   "{}" \;
find . -name "*activate*.*" -not -name "*.bak" -exec sed -i.bak "s|@TOOLS_DEACTIVATE@|${TOOLS_DEACTIVATE}|g" "{}" \;
find . -name "*activate*.*" -not -name "*.bak" -exec sed -i.bak "s|@CHOST@|${CHOST}|g"                     "{}" \;
find . -name "*activate*.*" -not -name "*.bak" -exec sed -i.bak "s|@LIBRARY_PREFIX@|${LIBRARY_PREFIX}|g"   "{}" \;
find . -name "*activate*.*" -not -name "*.bak" -exec sed -i.bak "s|@EXE_EXT@|${EXE_EXT}|g"                 "{}" \;

cat activate-binutils.sh
cat deactivate-binutils.sh

mkdir -p ${PREFIX}/etc/conda/{de,}activate.d
cp "${SRC_DIR}"/activate-binutils.sh ${PREFIX}/etc/conda/activate.d/activate-${PKG_NAME}.sh
cp "${SRC_DIR}"/deactivate-binutils.sh ${PREFIX}/etc/conda/deactivate.d/deactivate-${PKG_NAME}.sh

if [[ "$target_platform" == "win-"* ]]; then
  cp "${SRC_DIR}"/activate-binutils.bat ${PREFIX}/etc/conda/activate.d/activate-${PKG_NAME}.bat
  cp "${SRC_DIR}"/deactivate-binutils.bat ${PREFIX}/etc/conda/deactivate.d/deactivate-${PKG_NAME}.bat
  cp "${SRC_DIR}"/activate-binutils.ps1 ${PREFIX}/etc/conda/activate.d/activate-${PKG_NAME}.ps1
  cp "${SRC_DIR}"/deactivate-binutils.ps1 ${PREFIX}/etc/conda/deactivate.d/deactivate-${PKG_NAME}.ps1
fi
