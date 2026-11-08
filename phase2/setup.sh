#!/usr/bin/env bash
# Phase 2 setup: official L-NER files, format/offset checks and the BPE alignment ceiling per label.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -x .venv/bin/python ] || bash scripts/env.sh
source scripts/common.sh
need $BACKBONES
C=phase2/configs/n1_seq_full.yaml
[ -f data/phase2/Phase_2_LNER/Train/labels.txt ] || HF_HUB_OFFLINE=0 py ner.py "$C" download
py ner.py "$C" check
say "phase 2 setup done (logs/n1_seq_full.log)"
