
mkdir build
cd build

cmake %CMAKE_ARGS% -D CMAKE_INSTALL_PREFIX=%LIBRARY_PREFIX% ^
-D CMAKE_PREFIX_PATH=%LIBRARY_PREFIX% ^
-D CMAKE_BUILD_TYPE=Release ^
-D LIBXML2_WITH_ICU=%with_icu% ^
-D LIBXML2_WITH_LZMA=yes ^
-D LIBXML2_WITH_PYTHON=no ^
-D LIBXML2_WITH_SCHEMATRON=yes ^
-G "NMake Makefiles" ..
if errorlevel 1 exit 1

nmake
if errorlevel 1 exit 1

nmake install
if errorlevel 1 exit 1

copy %LIBRARY_LIB%\\libxml2.lib %LIBRARY_LIB%\\xml2.lib

setlocal EnableDelayedExpansion
for %%F in (activate deactivate) DO (
    if not exist %PREFIX%\etc\conda\%%F.d mkdir %PREFIX%\etc\conda\%%F.d
    copy %RECIPE_DIR%\%%F.bat %PREFIX%\etc\conda\%%F.d\%PKG_NAME%_%%F.bat
    copy %RECIPE_DIR%\%%F.ps1 %PREFIX%\etc\conda\%%F.d\%PKG_NAME%_%%F.ps1
    copy %RECIPE_DIR%\%%F.sh %PREFIX%\etc\conda\%%F.d\%PKG_NAME%_%%F.sh
)
