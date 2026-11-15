#!/usr/bin/env bash
# One command that rebuilds the forecasts and results from Pacific's pack.
#
#   bash reproduce.sh            # forecasts and test results from the accepted checkpoint (results/best.json)
#   MODE=base bash reproduce.sh  # the same with the untrained team (no checkpoint needed)
#   MODE=full bash reproduce.sh  # rerun the whole self-improvement loop first (hours on Tinker)
#
# Needs: Python 3.10+, the pack unpacked at data/raw/gpu-price-pack-v0.1.1-handover
# (or its zip at data/raw/), and TINKER_API_KEY in .env. No other data: Exa news is off
# unless data/events.jsonl exists, and every event in it is dated before each origin.
# Sampling uses fixed seeds (configs/team_v0.yaml), so reruns give the same forecasts.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
MODE="${MODE:-checkpoint}"
PACK=data/raw/gpu-price-pack-v0.1.1-handover
ZIP=data/raw/gpu-price-pack-v0.1.1-handover_pack-only.zip
ZIP_SHA=f0b52af28c52545732208bcc4905facd9b941aa5779da683e1d9884c283b7c7c

echo "== 1. the pack"
if [ ! -d "$PACK" ]; then
  [ -f "$ZIP" ] || { echo "put the pack zip at $ZIP (Track 2 Google Drive folder)"; exit 1; }
  echo "$ZIP_SHA  $ZIP" | shasum -a 256 -c -
  (cd data/raw && unzip -q "$(basename "$ZIP")")
fi
(cd "$PACK" && if command -v sha256sum >/dev/null; then sha256sum -c MANIFEST.sha256; else shasum -a 256 -c MANIFEST.sha256; fi) \
  | grep -v ': OK$' && { echo "pack files do not match MANIFEST.sha256"; exit 1; } || echo "pack intact"

echo "== 2. environment"
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q -e ".[dev]"
.venv/bin/python -m pytest -q

echo "== 3. self-improvement loop (MODE=full only)"
if [ "$MODE" = full ]; then
  RUN=outputs/rsi/reproduce
  .venv/bin/python scripts/rsi_loop.py --out "$RUN"
  cp "$RUN/best.json" results/best.json
  cp "$RUN/notebook.jsonl" "$RUN/archive.jsonl" results/ 2>/dev/null || true
fi

echo "== 4. locked test, ablations and the 6 Nov submission"
if [ "$MODE" = base ]; then
  .venv/bin/python scripts/final.py --base
else
  .venv/bin/python scripts/final.py --best results/best.json
fi

echo "== 5. the list-replay engine against Pacific's baselines (no LLM; validation and training origins only)"
.venv/bin/python scripts/engine_check.py

echo "== 6. results tables"
.venv/bin/python scripts/report.py
echo "done: submission/forecast_2026-11-06.csv, results/final.json, results/engine.md, results/report.md"
