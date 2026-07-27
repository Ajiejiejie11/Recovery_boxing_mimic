cmake -G "Ninja" -D EXPECTED_ENABLE_TESTS=OFF -D CMAKE_INSTALL_PREFIX=%LIBRARY_PREFIX% %SRC_DIR%
if errorlevel 1 exit 1

ninja install
if errorlevel 1 exit 1
