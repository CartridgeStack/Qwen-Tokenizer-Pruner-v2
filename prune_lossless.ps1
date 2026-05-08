# ─────────────────────────────────────────────────────────────────────────────
# prune_lossless.ps1
# Run from the repo root in a dedicated PowerShell window.
# ─────────────────────────────────────────────────────────────────────────────

$ErrorActionPreference = "Stop"

$old_model_path = ".\Qwen3.6-27B"
$new_model_path = ".\Qwen3.6-27B-pruned"
$support_data   = ".\sample_data"

# ── 1. Install dependencies (skip if already done) ────────────────────────────
Write-Host "==> Installing / verifying dependencies..." -ForegroundColor Cyan
pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet
pip install transformers safetensors langdetect tqdm --quiet

# ── 2. Run pruning ────────────────────────────────────────────────────────────
$prune_cmd = "python main.py " +
             "--old_model_path `"$old_model_path`" " +
             "--new_model_path `"$new_model_path`" " +
             "--support_data   `"$support_data`""

Write-Host ""
Write-Host "==> Running pruner:" -ForegroundColor Cyan
Write-Host $prune_cmd
Invoke-Expression $prune_cmd
if ($LASTEXITCODE -ne 0) {
    Write-Host "Pruning FAILED (exit $LASTEXITCODE)." -ForegroundColor Red
    exit $LASTEXITCODE
}

# ── 3. Lossless check ─────────────────────────────────────────────────────────
$check_cmd = "python check.py " +
             "--old_model_path `"$old_model_path`" " +
             "--new_model_path `"$new_model_path`" " +
             "--support_data   `"$support_data`""

Write-Host ""
Write-Host "==> Running lossless check:" -ForegroundColor Cyan
Write-Host $check_cmd
Invoke-Expression $check_cmd
if ($LASTEXITCODE -ne 0) {
    Write-Host "Lossless check FAILED (exit $LASTEXITCODE)." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "==> All done. Pruned model: $new_model_path" -ForegroundColor Green
