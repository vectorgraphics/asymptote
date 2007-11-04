@ECHO OFF
asy.exe %1
if %errorlevel% == 0 exit
echo.
PAUSE
