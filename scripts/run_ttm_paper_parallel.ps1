# BeeTrade — Paper test TTM song song V1 + V2 (JSONL trong reports/)
#
# Yêu cầu .env:
#   STRATEGY_ALGO=TTM
#   PAPER_MODE=true
#   DNSE_WS_BOARD_ID=G1       # TTM chỉ phái sinh
#
# Cách chạy (từ thư mục gốc project hoặc gọi trực tiếp file này):
#   .\scripts\run_ttm_paper_parallel.ps1
#   .\scripts\run_ttm_paper_parallel.ps1 -Symbol VN30F1M
#   .\scripts\run_ttm_paper_parallel.ps1 -Symbol VN30F1M -DebugFeed -DebugWsPipeline

param(
    [string] $Symbol = "VN30F1M",
    [switch] $DebugFeed,
    [switch] $DebugWsPipeline
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$pyArgs = @("scripts/paper_test.py", "--symbol", $Symbol)
if ($DebugFeed) { $pyArgs += "--debug-feed" }
if ($DebugWsPipeline) { $pyArgs += "--debug-ws-pipeline" }

Write-Host "[run_ttm_paper_parallel] python $($pyArgs -join ' ')" -ForegroundColor Cyan
& python @pyArgs
exit $LASTEXITCODE
