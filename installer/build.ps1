# Windows PowerShell build automation script for Music Mastery Enhancer
# This script prepares the environment, compiles the PyInstaller executable,
# and generates the guided Windows installer using Inno Setup (iscc).

$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "   Music Mastery Enhancer - Windows Build & Setup Script" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Check Python installation
Write-Host "[1/5] Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = & python --version 2>&1
    Write-Host "Found Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Error "Python is not installed or not in your PATH. Please install Python >= 3.11 from python.org."
    exit 1
}

# 2. Check and configure Virtual Environment
Write-Host "[2/5] Setting up virtual environment..." -ForegroundColor Yellow
$venvDir = Join-Path $PSScriptRoot "..\.venv"

if (Get-Command poetry -ErrorAction SilentlyContinue) {
    Write-Host "Found Poetry. Utilizing Poetry virtual environment..." -ForegroundColor Green
    
    # Try to enforce Python 3.11 for Poetry if py launcher is available
    if (Get-Command py -ErrorAction SilentlyContinue) {
        Write-Host "Configuring Poetry to use Python 3.11 via py launcher..." -ForegroundColor Gray
        & poetry env use py -3.11
    } else {
        & poetry env use python
    }

    Write-Host "Installing dependencies with Poetry..." -ForegroundColor Gray
    & poetry install --with dev
    & poetry run pip install pyinstaller
    # resemble-enhance's own metadata pins deepspeed==0.12.4 (fails to build on Windows) and
    # gradio==4.8.0 (only needed by its demo webapp). Install it without its declared deps —
    # pyproject.toml already lists the runtime deps it actually needs at inference time.
    & poetry run pip install resemble-enhance --no-deps
} else {
    Write-Host "Poetry not detected. Falling back to python venv..." -ForegroundColor Blue
    if (-not (Test-Path $venvDir)) {
        Write-Host "Creating local virtual environment in .venv..." -ForegroundColor Gray
        & python -m venv $venvDir
    }

    # Resolve scripts path for venv
    $pipPath = Join-Path $venvDir "Scripts\pip.exe"
    $pythonPath = Join-Path $venvDir "Scripts\python.exe"

    if (-not (Test-Path $pipPath)) {
        Write-Error "Failed to locate virtual environment pip at $pipPath"
        exit 1
    }

    Write-Host "Upgrading pip and installing dependencies..." -ForegroundColor Gray
    & $pipPath install --upgrade pip
    & $pipPath install -e ".[dev]"
    & $pipPath install pyinstaller
    # resemble-enhance's own metadata pins deepspeed==0.12.4 (fails to build on Windows) and
    # gradio==4.8.0 (only needed by its demo webapp). Install it without its declared deps —
    # pyproject.toml already lists the runtime deps it actually needs at inference time.
    & $pipPath install resemble-enhance --no-deps
}

# 3. Clean previous build artifacts
Write-Host "[3/5] Cleaning old build directories..." -ForegroundColor Yellow
$distDir = Join-Path $PSScriptRoot "..\dist"
$buildDir = Join-Path $PSScriptRoot "..\build"

if (Test-Path $distDir) {
    Write-Host "Removing $distDir..." -ForegroundColor Gray
    Remove-Item -Recurse -Force $distDir
}
if (Test-Path $buildDir) {
    Write-Host "Removing $buildDir..." -ForegroundColor Gray
    Remove-Item -Recurse -Force $buildDir
}

# 4. Build executable with PyInstaller
Write-Host "[4/5] Compiling application with PyInstaller..." -ForegroundColor Yellow
$specPath = Join-Path $PSScriptRoot "music_mastery_enhancer.spec"

if (-not (Test-Path $specPath)) {
    # Fallback if spec file is one level up
    $specPath = Join-Path $PSScriptRoot "..\installer\music_mastery_enhancer.spec"
}

if (Get-Command poetry -ErrorAction SilentlyContinue) {
    & poetry run pyinstaller $specPath --noconfirm
} else {
    $pyinstallerPath = Join-Path $venvDir "Scripts\pyinstaller.exe"
    if (Test-Path $pyinstallerPath) {
        & $pyinstallerPath $specPath --noconfirm
    } else {
        Write-Error "PyInstaller was not found in the virtual environment."
        exit 1
    }
}

$exePath = Join-Path $distDir "MusicMasteryEnhancer.exe"
if (Test-Path $exePath) {
    Write-Host "Successfully compiled executable: $exePath" -ForegroundColor Green
} else {
    Write-Error "Failed to produce MusicMasteryEnhancer.exe at $exePath."
    exit 1
}

# 5. Build installer with Inno Setup (ISCC)
Write-Host "[5/5] Compiling Inno Setup installer..." -ForegroundColor Yellow
$isccPath = ""

# Search standard locations safely
$cmdIscc = Get-Command iscc -ErrorAction SilentlyContinue
if ($cmdIscc) { $isccPath = $cmdIscc.Source }

if (-not $isccPath) {
    $standardPaths = @(
        "C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
        "C:\Program Files\Inno Setup 6\ISCC.exe",
        "C:\Program Files (x86)\Inno Setup 5\ISCC.exe",
        "C:\Program Files\Inno Setup 5\ISCC.exe"
    )
    foreach ($path in $standardPaths) {
        if (Test-Path $path) {
            $isccPath = $path
            break
        }
    }
}

$issPath = Join-Path $PSScriptRoot "setup_script.iss"

if ($isccPath) {
    Write-Host "Found Inno Setup compiler at: $isccPath" -ForegroundColor Green
    Write-Host "Compiling setup script: $issPath..." -ForegroundColor Gray
    & $isccPath $issPath

    $setupExe = Join-Path $PSScriptRoot "output\MusicMasteryEnhancer-Setup.exe"
    if (Test-Path $setupExe) {
        Write-Host "==========================================================" -ForegroundColor Green
        Write-Host "  Success! Guided installer created at:" -ForegroundColor Green
        Write-Host "  $setupExe" -ForegroundColor Green
        Write-Host "==========================================================" -ForegroundColor Green
    } else {
        Write-Warning "Inno Setup compilation completed, but could not locate the setup executable."
    }
} else {
    Write-Warning "Inno Setup (ISCC.exe) was not found in PATH or standard installation folders."
    Write-Warning "The single-file executable is built, but the installer setup wrapper could not be compiled."
    Write-Warning "Please download and install Inno Setup 6 (jrsoftware.org) and run:"
    Write-Warning "  iscc installer/setup_script.iss"
}