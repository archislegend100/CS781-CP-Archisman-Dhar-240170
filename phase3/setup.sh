#!/usr/bin/env bash
# Phase 3 setup: official rhetorical-role files and the format / label checks.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -x .venv/bin/python ] || bash scripts/env.sh
source scripts/common.sh
need $BACKBONES runs/selfrank_dpo/ckpt/best.pt
C=phase3/configs/base.yaml
HF_HUB_OFFLINE=0 py rr.py "$C" download
py rr.py "$C" check
say "phase 3 setup done (logs/base.log)"
