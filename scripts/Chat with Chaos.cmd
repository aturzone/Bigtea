@echo off
REM Double-click launcher: start the server, open the browser, done.
REM
REM The CLI is the real interface and stays the real interface. This exists
REM because "unpack a zip and get a chat window" is a different job from "learn
REM a flag", and someone who wants the second can still type it.
REM
REM Deliberately a .cmd and not a compiled .exe launcher: a second executable
REM would need signing, would trip Smart App Control on a fresh download, and
REM would hide the server's log -- which is the thing worth seeing when a model
REM takes forty seconds to load.

setlocal
cd /d "%~dp0"

if not defined CHAOS_MODELS set "CHAOS_MODELS=%USERPROFILE%\.chaos\models"
set "PORT=8080"

echo.
echo   C H A O S
echo.

if not exist "chaos-run.exe" (
  echo   chaos-run.exe is not next to this script.
  echo   Run this from the unpacked archive, where the .exe files live.
  echo.
  pause
  exit /b 1
)

REM Bare `chaos-run` lists what it can find. If nothing is listed there is no
REM point starting a server that cannot load anything.
chaos-run.exe >"%TEMP%\chaos-models.txt" 2>&1
findstr /C:"no models found" "%TEMP%\chaos-models.txt" >nul
if not errorlevel 1 (
  echo   No models found yet.
  echo.
  echo   Put a .gguf file here:
  echo     %CHAOS_MODELS%
  echo.
  echo   Chaos downloads nothing on its own -- models are yours to obtain,
  echo   under their own licences. Then run this again.
  echo.
  start "" "%CHAOS_MODELS%"
  pause
  exit /b 1
)

echo   Starting the server. The first load can take a while on a big model;
echo   this window shows what it is doing. Close it to stop.
echo.
echo   Opening http://127.0.0.1:%PORT% in your browser.
echo.

REM The browser is opened before the server is ready on purpose: the page is a
REM static asset served the moment the socket binds, and a tab that is already
REM waiting beats one the user has to remember to open.
start "" "http://127.0.0.1:%PORT%"

REM No model named: chaos-serve picks the only one, or lists them and exits.
chaos-serve.exe --port %PORT%

echo.
echo   The server has stopped.
pause
