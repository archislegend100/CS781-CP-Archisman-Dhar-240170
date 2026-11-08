#!/usr/bin/env bash
# Phase 6 setup: official summarization files, chunk-aligned tokenisation for each backbone family, and
# (optional) roberta-large for held-out selection with the official BERTScore (S18-S20 only; never inside a model).
set -euo pipefail
cd "$(dirname "$0")/.."
[ -x .venv/bin/python ] || bash scripts/env.sh
source scripts/common.sh
need $BACKBONES
HF_HUB_OFFLINE=0 py sm.py phase6/configs/base.yaml download
for c in s1_gpt_lorma s4_seq_lora s3_long_qlora; do py sm.py "phase6/configs/$c.yaml" prep; done
if [ ! -f models/roberta-large/config.json ]; then
  HF_HUB_OFFLINE=0 python -c "from huggingface_hub import snapshot_download as d; d('FacebookAI/roberta-large', local_dir='models/roberta-large', allow_patterns=['*.json', '*.txt', 'model.safetensors'])" \
    >> logs/roberta.log 2>&1 || say "roberta-large not downloaded: S18-S20 fall back to the Phase-1 proxy scorer"
fi
say "phase 6 setup done"
