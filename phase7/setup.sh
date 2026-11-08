#!/usr/bin/env bash
# Phase 7 setup: official PCR queries / candidate pools, tokenisation, and the BM25 (uni+bigram) index.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -x .venv/bin/python ] || bash scripts/env.sh
source scripts/common.sh
need $BACKBONES
C=phase7/configs/base.yaml
HF_HUB_OFFLINE=0 py pcr.py "$C" download
py pcr.py "$C" prep
[ -f runs/pcr/bm25.pt ] || py pcr.py "$C" bm25
say "phase 7 setup done"
