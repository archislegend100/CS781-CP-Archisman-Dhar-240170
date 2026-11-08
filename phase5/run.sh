#!/usr/bin/env bash
# Phase 5 (statute identification): every submission L1-L29 in order on one GPU; the final system is L29.
# Usage: CUDA_VISIBLE_DEVICES=0 bash phase5/run.sh          bash phase5/run.sh one lsi.py l1_gpt_lorma train
PHASE=phase5
source "$(dirname "$0")/../scripts/common.sh"
[ "${1:-}" = one ] && { shift; one "$@"; exit 0; }
need $BACKBONES runs/lsi/tok/train.pt runs/lsi/tok/dev.pt runs/lsi/tok/test.pt
cfg() { echo "$PHASE/configs/$1.yaml"; }

run() {   # single model: train (LWAN head + ASL, step chosen on dev) + per-statute thresholds + prediction
  local c; c=$(cfg "$1"); done_zip "$c" && return 0
  py lsi.py "$c" train; py lsi.py "$c" predict
}
step() {  # step <mode> <config>: predict (chunk aggregation) / ensemble / stack / devjoin
  local c; c=$(cfg "$2"); done_zip "$c" && return 0
  py lsi.py "$c" "$1"
}
devfold() {  # devfold <base config> <run prefix> <id> [key=value ...]: two dev cross-fitted members
  local b=$1 p=$2 id=$3 k n; shift 3
  for k in 0 1; do
    n="${p}_df$k"; [ -f "runs/lsi/$n/probs.pt" ] && continue
    py lsi.py "$(cfg "$b")" train "name=$n" "id=${id}F$k" "lsi.devfold=$k" "$@"
    py lsi.py "$(cfg "$b")" predict "name=$n" "id=${id}F$k" "lsi.devfold=$k" "$@"
  done
}

# Round 1: one technique per architecture, ensemble, KD student, weighted ensemble, BCE + mean-pool variant
for m in l1_gpt_lorma l2_hyb_adapter l5_gpt_full l6_gpt_partial l3_long_qlora l4_seq_adalora l7_gpt_prompt; do run "$m"; done
step ensemble l8_ens
[ -f runs/lsi/teacher.pt ] || py lsi.py "$(cfg l9_kd)" teach
run l9_kd
step ensemble l10_ens_final
run l11_gpt_lorma_pool_bce
# Round 2: chunk aggregation at prediction time (max over 3 windows), ensembles and stacking
for m in l12_gpt_full_chunk l14_kd_chunk l13_hyb_adapter_chunk l15_gpt_lorma_chunk l17_gpt_partial_chunk l16_long_qlora_chunk; do
  step predict "$m"
done
step ensemble l18_ens_chunk; step stack l20_stack_orig; step stack l19_stack
# Round 3: dev cross-fitting (KD from a dev-aware teacher, adapters, full fine-tuning), ensemble, stacking
[ -f runs/lsi/teacher_dev.pt ] || py lsi.py "$(cfg l21_teach)" teach
C=(lsi.trainprobs=false lsi.chunks=3 lsi.agg=max)
devfold l9_kd lsi_kd_dcv L21 "${C[@]}" lsi.kd.teacher=runs/lsi/teacher_dev.pt; step devjoin l21_kd_dcv
devfold l2_hyb_adapter lsi_hyb_adapter_dcv L23 "${C[@]}";                     step devjoin l23_hyb_adapter_dcv
devfold l5_gpt_full lsi_gpt_full_dcv L22 "${C[@]}";                           step devjoin l22_gpt_full_dcv
step ensemble l24_ens_dcv; step stack l25_stack_dcv
# Round 4: multiple-instance chunk training (L27), distribution-balanced loss (L26), final ensembles L28 / L29
devfold l9_kd lsi_kd_mil_dcv L27 lsi.trainprobs=false lsi.kd.teacher=runs/lsi/teacher_dev.pt lsi.milchunks=3 lsi.chunks=1 lsi.batch=2 lsi.accum=8
step devjoin l27_kd_mil_dcv
step ensemble l28_ens_final
devfold l2_hyb_adapter lsi_hyb_adapter_db_dcv L26 lsi.trainprobs=false lsi.loss=db lsi.chunks=3 lsi.agg=max
step devjoin l26_hyb_adapter_db_dcv
step ensemble l29_ens_final
say "phase 5 finished; final submission: $(zipof "$(cfg l29_ens_final)")"
