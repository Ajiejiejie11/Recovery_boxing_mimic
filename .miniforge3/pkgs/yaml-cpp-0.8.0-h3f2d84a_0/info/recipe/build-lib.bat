IF not x%PKG_NAME:static=%==x%PKG_NAME% (
    set SHARED="OFF"
) ELSE (
    set SHARED="ON"
)

cmake -B build-%PKG_NAME% ^
    -G Ninja ^
    -D CMAKE_MSVC_RUNTIME_LIBRARY="MultiThreadedDLL" ^
    -D BUILD_SHARED_LIBS=%SHARED% ^
    -D YAML_BUILD_SHARED_LIBS=%SHARED% ^
    -D YAML_CPP_BUILD_TESTS=OFF ^
    -D YAML_MSVC_SHARED_RT=ON ^
    %CMAKE_ARGS%
if errorlevel 1 exit 1

cmake --build build-%PKG_NAME% --parallel %CPU_COUNT% --verbose
if errorlevel 1 exit 1

cmake --install build-%PKG_NAME%
if errorlevel 1 exit 1
