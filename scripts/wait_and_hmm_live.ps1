# Chờ N phút rồi chạy một bước HMM live (--submit).
# Dùng khi phiên sắp mở: ví dụ phiên sau ~50 phút.
#
#   powershell -ExecutionPolicy Bypass -File scripts/wait_and_hmm_live.ps1
#   powershell -ExecutionPolicy Bypass -File scripts/wait_and_hmm_live.ps1 -Minutes 50 -Symbol VN30F1M

param(
    [int]$Minutes = 50,
    [string]$Symbol = "VN30F1M"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "BeeTrade: cho $Minutes phut (PAPER_MODE doc tu .env)..."
Write-Host "  Thu muc: $root"
Start-Sleep -Seconds ($Minutes * 60)

Write-Host "Chay: python scripts/hmm_live.py --symbol $Symbol --submit"
python scripts/hmm_live.py --symbol $Symbol --submit
