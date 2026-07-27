cmake -B build ^
    -G "Ninja" ^
    -D CMAKE_MSVC_RUNTIME_LIBRARY="MultiThreadedDLL" ^
    -D CMAKE_C_USE_RESPONSE_FILE_FOR_OBJECTS=ON ^
    -D BUILD_SHARED_LIBS=OFF ^
    -D CMAKE_RELEASE_POSTFIX="_static" ^
    -D CMAKE_FIND_LIBRARY_SUFFIXES="static.lib" ^
    -D ENABLE_ZLIB=ON ^
    -D ZLIB_USE_STATIC_LIBS=ON ^
    -D ENABLE_BZip2=ON ^
    -D BZIP2_LIBRARIES="%LIBRARY_LIB%\\bzip2_static.lib" ^
    -D ENABLE_ZSTD=ON ^
    -D ZSTD_LIBRARY="%LIBRARY_LIB%\\zstd_static.lib" ^
    -D ENABLE_LZMA=OFF ^
    -D ENABLE_LZO=OFF ^
    -D ENABLE_CNG=OFF ^
    -D ENABLE_OPENSSL=OFF ^
    -D ENABLE_NETTLE=OFF ^
    -D ENABLE_LIBXML2=OFF ^
    -D ENABLE_EXPAT=OFF ^
    %CMAKE_ARGS%
if %errorlevel% NEQ 0 exit /b %errorlevel%

cmake --build build --parallel %CPU_COUNT% --verbose
if %errorlevel% NEQ 0 exit /b %errorlevel%

cmake --install build
if %errorlevel% NEQ 0 exit /b %errorlevel%
