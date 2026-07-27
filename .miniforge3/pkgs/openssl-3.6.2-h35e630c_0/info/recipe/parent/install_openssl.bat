@echo on
setlocal enabledelayedexpansion

nmake install
if %ERRORLEVEL% neq 0 exit 1

:: don't include html docs that get installed
rd /s /q %LIBRARY_PREFIX%\html

:: install pkgconfig metadata (useful for downstream packages);
:: adapted from inspecting the conda-forge .pc files for unix, as well as
:: https://github.com/microsoft/vcpkg/blob/master/ports/openssl/install-pc-files.cmake
mkdir %LIBRARY_PREFIX%\lib\pkgconfig
for %%F in (openssl libssl libcrypto) DO (
    echo prefix=%LIBRARY_PREFIX:\=/% > %%F.pc
    type %RECIPE_DIR%\win_pkgconfig\%%F.pc.in >> %%F.pc
    echo Version: %PKG_VERSION% >> %%F.pc
    copy %%F.pc %LIBRARY_PREFIX%\lib\pkgconfig\%%F.pc
)

mkdir %LIBRARY_PREFIX%\ssl\certs
type NUL > %LIBRARY_PREFIX%\ssl\certs\.keep

:: Copy the [de]activate scripts to %PREFIX%\etc\conda\[de]activate.d.
:: This will allow them to be run on environment activation.
for %%F in (activate deactivate) DO (
    if not exist %PREFIX%\etc\conda\%%F.d mkdir %PREFIX%\etc\conda\%%F.d
    copy "%RECIPE_DIR%\%%F-win.bat" "%PREFIX%\etc\conda\%%F.d\%PKG_NAME%_%%F-win.bat"
    copy "%RECIPE_DIR%\%%F-win.ps1" "%PREFIX%\etc\conda\%%F.d\%PKG_NAME%_%%F-win.ps1"
    :: Copy unix shell activation scripts, needed by Windows Bash users
    copy "%RECIPE_DIR%\%%F-win.sh" "%PREFIX%\etc\conda\%%F.d\%PKG_NAME%_%%F-win.sh"
)
