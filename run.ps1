$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not (Test-Path '.venv')) { py -3.11 -m venv .venv }
& .\.venv\Scripts\python.exe -m pip install --upgrade pip wheel
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
Write-Host "ShortForge V2 dependencies installed. GPU CUDA libraries for faster-whisper are configured separately; see README.md."
