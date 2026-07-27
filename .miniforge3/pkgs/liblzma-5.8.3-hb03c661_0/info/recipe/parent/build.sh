#!/bin/bash

./configure --prefix=${PREFIX}  \
            --build=${BUILD}    \
            --host=${HOST}

make -j${CPU_COUNT} ${VERBOSE_AT}

if [[ "$CONDA_BUILD_CROSS_COMPILATION" != 1 ]]; then
  make check || cat tests/test-suite.log
fi

# remove libtool files
find $PREFIX -name '*.la' -delete
