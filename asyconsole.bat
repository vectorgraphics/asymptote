@ECHO OFF
"C:\Program Files\Asymptote\asy.exe" %1
if %errorlevel% == 0 exit
echo.
PAUSE
