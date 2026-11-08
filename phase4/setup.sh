#!/usr/bin/env bash
# Phase 4 setup: official CJPE files and BPE tokenisation of train / dev / test (runs/cj/tok).
set -euo pipefail
cd "$(dirname "$0")/.."
[ -x .venv/bin/python ] || bash scripts/env.sh
source scripts/common.sh
need $BACKBONES
C=phase4/configs/base.yaml
HF_HUB_OFFLINE=0 py cj.py "$C" download
py cj.py "$C" prep
say "phase 4 setup done"
