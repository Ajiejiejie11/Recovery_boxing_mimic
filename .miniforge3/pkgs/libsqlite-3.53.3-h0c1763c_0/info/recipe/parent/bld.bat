@echo on

:: Define common options
set OPTIONS=-DSQLITE_DQS=3 ^
            -DSQLITE_ENABLE_COLUMN_METADATA ^
            -DSQLITE_ENABLE_DBSTAT_VTAB ^
            -DSQLITE_ENABLE_DESERIALIZE ^
            -DSQLITE_ENABLE_EXPLAIN_COMMENTS ^
            -DSQLITE_ENABLE_FTS3 ^
            -DSQLITE_ENABLE_FTS3_PARENTHESIS ^
            -DSQLITE_ENABLE_FTS3_TOKENIZER ^
            -DSQLITE_ENABLE_FTS4 ^
            -DSQLITE_ENABLE_FTS5 ^
            -DSQLITE_ENABLE_GEOPOLY ^
            -DSQLITE_ENABLE_JSON1 ^
            -DSQLITE_ENABLE_MATH_FUNCTIONS ^
            -DSQLITE_ENABLE_PREUPDATE_HOOK ^
            -DSQLITE_ENABLE_RTREE ^
            -DSQLITE_ENABLE_SESSION ^
            -DSQLITE_ENABLE_STAT4 ^
            -DSQLITE_ENABLE_STMTVTAB ^
            -DSQLITE_ENABLE_UNLOCK_NOTIFY ^
            -DSQLITE_ENABLE_UPDATE_DELETE_LIMIT ^
            -DSQLITE_LIKE_DOESNT_MATCH_BLOBS ^
            -DSQLITE_MAX_EXPR_DEPTH=10000 ^
            -DSQLITE_MAX_VARIABLE_NUMBER=250000 ^
            -DSQLITE_SOUNDEX ^
            -DSQLITE_STRICT_SUBTYPE=1 ^
            -DSQLITE_THREADSAFE=1 ^
            -DSQLITE_USE_URI ^
            -DHAVE_ISNAN

:: Build DLL and shell executable
nmake /f Makefile.msc DYNAMIC_SHELL=1 ^
                      USE_NATIVE_LIBPATHS=1 ^
                      OPT_FEATURE_FLAGS="%OPTIONS%"
if %ERRORLEVEL% neq 0 exit 1


COPY sqlite3.exe  %LIBRARY_BIN% || exit 1
COPY sqlite3.dll  %LIBRARY_BIN% || exit 1
COPY sqlite3.lib  %LIBRARY_LIB% || exit 1
COPY sqlite3.h    %LIBRARY_INC% || exit 1
COPY sqlite3ext.h %LIBRARY_INC% || exit 1

if not exist %LIBRARY_LIB%\pkgconfig mkdir %LIBRARY_LIB%\pkgconfig || exit 1

if "%PKG_VERSION%"=="" (
  echo PKG_VERSION is not set.
  exit 1
)

if not exist sqlite3.pc.in (
  echo Missing sqlite3.pc.in in source directory.
  exit 1
)

powershell -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass ^
  -Command "$pc = Get-Content -Raw 'sqlite3.pc.in';" ^
           "$pc = $pc -replace '@prefix@','${pcfiledir}/../..';" ^
           "$pc = $pc -replace '@exec_prefix@','${prefix}';" ^
           "$pc = $pc -replace '@libdir@','${prefix}/lib';" ^
           "$pc = $pc -replace '@includedir@','${prefix}/include';" ^
           "$pc = $pc -replace '@PACKAGE_VERSION@','%PKG_VERSION%';" ^
           "$pc = $pc -replace '@LDFLAGS_MATH@','';" ^
           "$pc = $pc -replace '@LDFLAGS_ZLIB@','';" ^
           "$pc = $pc -replace '@LDFLAGS_DLOPEN@','';" ^
           "$pc = $pc -replace '@LDFLAGS_PTHREAD@','';" ^
           "$pc = $pc -replace '@LDFLAGS_ICU@','';" ^
           "Set-Content -Path '%LIBRARY_LIB%\\pkgconfig\\sqlite3.pc' -Value $pc -NoNewline"
if %ERRORLEVEL% neq 0 exit 1
