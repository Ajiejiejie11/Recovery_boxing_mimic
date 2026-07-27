#!/bin/bash

set -euxo pipefail

# Prevent running ldconfig when cross-compiling.
if [[ "${BUILD}" != "${HOST}" ]]; then
  echo "#!/usr/bin/env bash" > ldconfig
  chmod +x ldconfig
  export PATH=${PWD}:$PATH
fi

export OPTIONS="-DSQLITE_DQS=3 \
                -DSQLITE_ENABLE_COLUMN_METADATA \
                -DSQLITE_ENABLE_DBSTAT_VTAB \
                -DSQLITE_ENABLE_DESERIALIZE \
                -DSQLITE_ENABLE_EXPLAIN_COMMENTS \
                -DSQLITE_ENABLE_FTS3 \
                -DSQLITE_ENABLE_FTS3_PARENTHESIS \
                -DSQLITE_ENABLE_FTS3_TOKENIZER \
                -DSQLITE_ENABLE_FTS4 \
                -DSQLITE_ENABLE_FTS5 \
                -DSQLITE_ENABLE_GEOPOLY \
                -DSQLITE_ENABLE_JSON1 \
                -DSQLITE_ENABLE_MATH_FUNCTIONS \
                -DSQLITE_ENABLE_PREUPDATE_HOOK \
                -DSQLITE_ENABLE_RTREE \
                -DSQLITE_ENABLE_SESSION \
                -DSQLITE_ENABLE_STAT4 \
                -DSQLITE_ENABLE_STMTVTAB \
                -DSQLITE_ENABLE_UNLOCK_NOTIFY \
                -DSQLITE_ENABLE_UPDATE_DELETE_LIMIT \
                -DSQLITE_LIKE_DOESNT_MATCH_BLOBS \
                -DSQLITE_MAX_EXPR_DEPTH=10000 \
                -DSQLITE_MAX_VARIABLE_NUMBER=250000 \
                -DSQLITE_SOUNDEX \
                -DSQLITE_STRICT_SUBTYPE=1 \
                -DSQLITE_THREADSAFE=1 \
                -DSQLITE_USE_URI \
                -DHAVE_ISNAN"

if [[ $target_platform =~ linux.* ]]; then
    export CFLAGS="${CFLAGS} -DHAVE_PREAD64 -DHAVE_PWRITE64"
    export SONAME_SWITCH="--soname=legacy"
else
    export SONAME_SWITCH=""
fi

if [[ "$target_platform" == "linux-ppc64le" ]]; then
    export PPC64LE="--build=ppc64le-linux"
else
    export PPC64LE=""
fi

if [[ "${with_icu}" == "true" ]]; then
    export ICU_FLAGS="--with-icu-config=${PREFIX}/bin/icu-config --enable-icu-collations"
else
    export ICU_FLAGS=""
fi

./configure --prefix=${PREFIX} \
            --build=${BUILD} \
            --host=${HOST} \
            --enable-threadsafe \
            --enable-load-extension \
            --disable-static \
            --disable-static-shell \
            --with-readline-header="${PREFIX}/include/readline/readline.h" \
            ${SONAME_SWITCH} \
            ${ICU_FLAGS} \
            CFLAGS="${CFLAGS} ${OPTIONS} -I${PREFIX}/include" \
            LDFLAGS="${LDFLAGS} -L${PREFIX}/lib" \
            ${PPC64LE}

make -j${CPU_COUNT}
make install

# We can remove this when we start using the new conda-build.
find $PREFIX -name '*.la' -delete
