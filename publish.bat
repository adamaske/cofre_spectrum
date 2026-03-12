@echo off
setlocal
set REPO_DIR=%~dp0

REM Read token from file at repo root
set TOKEN_FILE=%REPO_DIR%pypi_token.txt
if not exist "%TOKEN_FILE%" (
    echo ERROR: Token file not found at %TOKEN_FILE%
    exit /b 1
)

REM Use for /f to read token
set PYPI_TOKEN=
for /f "usebackq delims=" %%T in ("%TOKEN_FILE%") do (
    if not defined PYPI_TOKEN set PYPI_TOKEN=%%T
)
if not defined PYPI_TOKEN (
    echo ERROR: Token file is empty.
    exit /b 1
)
echo Token loaded OK.

REM Ensure build tools are installed
echo Installing build tools...
call python -m pip install --quiet build twine
if errorlevel 1 goto :pipfail
goto :pipdone
:pipfail
echo ERROR: pip install failed.
exit /b 1
:pipdone

REM Clean old build artifacts
echo Cleaning old dist...
if exist "%REPO_DIR%dist" rmdir /s /q "%REPO_DIR%dist"
if exist "%REPO_DIR%src\cofre_spectrum.egg-info" rmdir /s /q "%REPO_DIR%src\cofre_spectrum.egg-info"

REM Build
echo Building package...
cd /d "%REPO_DIR%"
call python -m build
if errorlevel 1 goto :buildfail
goto :builddone
:buildfail
echo ERROR: Build failed.
exit /b 1
:builddone

REM Upload
echo Uploading to PyPI...
call python -m twine upload "%REPO_DIR%dist\*" -u __token__ -p "%PYPI_TOKEN%"
if errorlevel 1 goto :uploadfail
goto :uploaddone
:uploadfail
echo ERROR: Upload failed.
exit /b 1
:uploaddone

echo Done! Package published successfully.
endlocal