@echo off
setlocal
rem Prescription map system launcher
chcp 65001 >nul

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

set "PYTHON_EXE=D:\miniconda3\envs\farm_edge\python.exe"

if not exist "%PYTHON_EXE%" (
  echo Python env not found: %PYTHON_EXE%
  echo Trying fallback: python on PATH...
  set "PYTHON_EXE=python"
)

"%PYTHON_EXE%" -c "import numpy, rasterio, flask, PIL" 2>nul
if errorlevel 1 (
  echo Missing dependencies. Please install: numpy rasterio pillow flask
  pause
  exit /b 1
)

"%PYTHON_EXE%" app.py
pause
