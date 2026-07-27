@echo on

REM use VC-WIN64-CLANGASM-ARM on arm64 to use clang-cl
REM from Visual Studio to enable assembly instructions
REM and uplink support. See NOTES-WINDOWS.md in source.
if "%target_platform%"=="win-arm64" (
    set OSSL_CONFIGURE=VC-WIN64-CLANGASM-ARM
) else if "%ARCH%"=="32" (
    set OSSL_CONFIGURE=VC-WIN32
) else (
    set OSSL_CONFIGURE=VC-WIN64A
)

REM Configure step
REM
REM Conda currently does not perform prefix replacement on Windows, so
REM OPENSSLDIR cannot (reliably) be used to provide functionality such as a
REM default configuration and standard CA certificates on a per-environment
REM basis.  Given that, we set OPENSSLDIR to a location with extremely limited
REM write permissions to limit the risk of non-privileged users exploiting
REM OpenSSL's engines feature to perform arbitrary code execution attacks
REM against applications that load the OpenSSL DLLs.
REM
REM On top of that, we also set the SSL_CERT_FILE environment variable
REM via an activation script to point to the ca-certificates provided CA root file.
REM
REM Copied from AnacondaRecipes/openssl-feedstock
perl configure %OSSL_CONFIGURE%   ^
    --prefix=%LIBRARY_PREFIX%     ^
    --openssldir="%CommonProgramFiles%\ssl" ^
    enable-legacy                 ^
    no-fips                       ^
    no-module                     ^
    no-zlib                       ^
    shared
if %ERRORLEVEL% neq 0 exit 1

REM specify in metadata where the packaging is coming from
set "OPENSSL_VERSION_BUILD_METADATA=+conda_forge"

REM Build step
nmake
if %ERRORLEVEL% neq 0 exit 1

REM Testing step
if NOT "%CONDA_BUILD_CROSS_COMPILATION%" == "1" (
    nmake test
    if %ERRORLEVEL% neq 0 exit 1
)
