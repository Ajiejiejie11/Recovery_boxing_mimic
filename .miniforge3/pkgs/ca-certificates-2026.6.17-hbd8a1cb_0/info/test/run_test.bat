

@echo on

test -f "${PREFIX}/ssl/cacert.pem"
IF %ERRORLEVEL% NEQ 0 exit /B 1
test -f "${PREFIX}/ssl/cert.pem"
IF %ERRORLEVEL% NEQ 0 exit /B 1
curl --cacert "${PREFIX}/ssl/cacert.pem" https://www.google.com
IF %ERRORLEVEL% NEQ 0 exit /B 1
exit /B 0
