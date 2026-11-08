#!/usr/bin/env bash
# Phase 1 setup: environment, official Phase-1 data, document-level split, token caches, dev set, GPU checks.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -x .venv/bin/python ] || bash scripts/env.sh
source scripts/common.sh
C=phase1/configs/legalgpt.yaml
HF_HUB_OFFLINE=0 py data.py "$C" download
[ -f splits/meta.json ] || py data.py "$C" split
py cache.py "$C" build "data.workers=$(nproc)"
[ -f data/dev/truth.json ] || py validate.py "$C" devset 500
logged remotecheck python src/remotecheck.py gpu ckpt data gen
say "phase 1 setup done"
