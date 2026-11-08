#!/usr/bin/env bash
# Phase 5 setup: official LSI files (facts, statute texts, labels) and BPE tokenisation (runs/lsi/tok).
set -euo pipefail
cd "$(dirname "$0")/.."
[ -x .venv/bin/python ] || bash scripts/env.sh
source scripts/common.sh
need $BACKBONES
C=phase5/configs/base.yaml
HF_HUB_OFFLINE=0 py lsi.py "$C" download
py lsi.py "$C" prep
say "phase 5 setup done"
