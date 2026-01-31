#!/usr/bin/env pwsh

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "CHANGI AEROVISION - CNN DEMO - AUTOMATIC SETUP" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists
if (-not (Test-Path ".venv")) {
    Write-Host "[1/3] Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
    Write-Host "    ✓ Virtual environment created successfully!" -ForegroundColor Green
    Write-Host ""
} else {
    Write-Host "[1/3] Virtual environment already exists" -ForegroundColor Green
    Write-Host ""
}

# Activate virtual environment
Write-Host "[2/3] Activating virtual environment..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1

# Install dependencies
Write-Host "[3/3] Installing dependencies..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
Write-Host "    ✓ Dependencies installed successfully!" -ForegroundColor Green
Write-Host ""

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "SETUP COMPLETE! Starting CNN Demo..." -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "The web interface will open at: http://localhost:5000" -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Run the demo
python CNN_demo.py
