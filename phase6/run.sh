#!/usr/bin/env bash
# Phase 6 (summarization): every submission S1-S20 in order on one GPU; the final system is S13.
# Usage: CUDA_VISIBLE_DEVICES=0 bash phase6/run.sh          bash phase6/run.sh one sm.py s1_gpt_lorma train
PHASE=phase6
source "$(dirname "$0")/../scripts/common.sh"
[ "${1:-}" = one ] && { shift; one "$@"; exit 0; }
need $BACKBONES
cfg() { echo "$PHASE/configs/$1.yaml"; }

run() {   # chunk-aligned summarizer: train + held-out length budget + test summaries
  local c; c=$(cfg "$1"); done_zip "$c" && return 0
  py sm.py "$c" train; py sm.py "$c" predict
}
step() {  # step <mode> <config>: predict / mbr / hybrid / hybrid2 / hybrid3
  local c; c=$(cfg "$2"); done_zip "$c" && return 0
  py sm.py "$c" "$1"
}

# Round 1: one technique per architecture, lead-only ablation, KD student, document-level MBR (S8)
run s1_gpt_lorma; step predict s9_lead_ablation
for m in s4_seq_lora s2_hyb_houlsby s5_gpt_pfeiffer s6_gpt_adalora s7_kd s3_long_qlora; do run "$m"; done
[ -f runs/sm/sm_mbr/summaries.pt ] || py sm.py "$(cfg s8_mbr)" mbr
# Round 2: abstract-then-extract fill to a held-out word budget (S13 = fill on the S8 summaries, final)
for m in s10_hyb_s1 s11_hyb_s2 s12_hyb_s3 s13_hyb_mbr; do step hybrid "$m"; done
step mbr s14_mbr_hybrid
# Round 3: budgets chosen with a Phase-1 embedding proxy of BERTScore's 512-token window
for m in s15_proxy_mbr s17_proxy_mbr_abs s16_proxy_s1; do step hybrid2 "$m"; done
# Round 4: chunk-interleaved fill (S18-S20), selected with the official scorer when roberta-large is present
if [ -f models/roberta-large/config.json ] && python -c "import bert_score" 2>/dev/null; then
  py sm.py "$(cfg s18_win_s1)" bsscan
fi
for m in s18_win_s1 s19_win_s2 s20_win_s3; do step hybrid3 "$m"; done
say "phase 6 finished; final submission: $(zipof "$(cfg s13_hyb_mbr)")"
