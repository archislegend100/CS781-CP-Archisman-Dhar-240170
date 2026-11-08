#!/usr/bin/env bash
# Phase 7 (prior case retrieval): every submission P1-P30 in order on one GPU; the final system is P26.
# Usage: CUDA_VISIBLE_DEVICES=0 bash phase7/run.sh          bash phase7/run.sh one pcr.py p1_gpt_lorma train
PHASE=phase7
source "$(dirname "$0")/../scripts/common.sh"
[ "${1:-}" = one ] && { shift; one "$@"; exit 0; }
need $BACKBONES runs/pcr/bm25.pt
cfg() { echo "$PHASE/configs/$1.yaml"; }

run() {   # dense retriever (InfoNCE, BM25 hard negatives): train + dev/test retrieval
  local c; c=$(cfg "$1"); done_zip "$c" && return 0
  py pcr.py "$c" train; py pcr.py "$c" predict
}
step() {  # step <script> <mode> <config>
  local c; c=$(cfg "$3"); done_zip "$c" && return 0
  py "$1" "$c" "$2"
}

# Round 1: one technique per architecture, KD student, fusion with BM25 / TF-IDF (P8)
for m in p1_gpt_lorma p7_kd p5_gpt_full p2_hyb_ia3 p4_seq_partial p3_long_qlora p6_gpt_prompt; do run "$m"; done
step pcr.py fuse p8_fuse
# Round 2: hubness-corrected re-ranking with query expansion (P9, P10)
py pcr2.py "$(cfg p9_rerank)" emb
step pcr2.py rerank p9_rerank; step pcr2.py rerank p10_rerank3
# Round 3: cross-fitted listwise learning-to-rank (P12, P14), second LoRMA+ retriever (P11)
step pcr3.py ltr p12_ltr
run p11_gpt_lorma2
py pcr2.py "$(cfg p13_rerank2)" emb
step pcr2.py rerank p13_rerank2; step pcr3.py ltr p14_ltr2
# Round 4: chunk MaxSim and query-graph features (P15, P16), multiple-instance retrievers (P17, P19)
py pcr4.py "$(cfg p15_ltr3)" chunkemb
step pcr4.py ltr p15_ltr3; step pcr4.py ltr p16_ltr3b
run p17_gpt_mil; step pcr4.py ltr p18_ltr4
run p19_hyb_mil; step pcr4.py ltr p20_ltr5
step pcr4.py ltr p21_ltr_var1; step pcr4.py ltr p22_ltr_var2
step pcr4.py ens p23_ens; step pcr4.py ens p24_ens_final
# Round 5: chunk-pair cross-encoder (P25) as an extra ranker view (P26 = final), dev-tuned ensembles (P27, P30)
c=$(cfg p25_ce)
if ! done_zip "$c"; then py pcr5.py "$c" prep; py pcr5.py "$c" train; py pcr5.py "$c" predict; fi
step pcr4.py ltr p26_ltr_ce; step pcr4.py ens p27_ens_ce
c=$(cfg p28_ce2)
if ! done_zip "$c"; then  # training may stop with an error after best.pt is written; continue from best.pt
  say "python src/pcr5.py $c train"
  [ -n "${DRYRUN:-}" ] || python src/pcr5.py "$c" train >> logs/p28_ce2.log 2>&1 || say "p28 training ended with an error (logs/p28_ce2.log)"
  need runs/pcr/pcr_ce2/best.pt; py pcr5.py "$c" predict
fi
step pcr4.py ltr p29_ltr_ce2; step pcr4.py ens p30_ens_final
say "phase 7 finished; final submission: $(zipof "$(cfg p26_ltr_ce)")"
