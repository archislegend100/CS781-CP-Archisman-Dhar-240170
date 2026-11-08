#!/usr/bin/env bash
# Phase 4 (CJPE): every submission C1-C27 in order on one GPU. C23 scored best but trained on dev cases that
# are also test cases (see README); C19 is the best system trained without dev labels.
# Usage: CUDA_VISIBLE_DEVICES=0 bash phase4/run.sh          bash phase4/run.sh one cj.py c1_gpt_lorma train
PHASE=phase4
source "$(dirname "$0")/../scripts/common.sh"
[ "${1:-}" = one ] && { shift; one "$@"; exit 0; }
need $BACKBONES runs/cj/tok/train.pt runs/cj/tok/dev.pt runs/cj/tok/test.pt
cfg() { echo "$PHASE/configs/$1.yaml"; }

run() {   # single model on the final 2,048 tokens: train (step + threshold chosen on dev) + prediction
  local c; c=$(cfg "$1"); done_zip "$c" && return 0
  py cj.py "$c" train; py cj.py "$c" predict
}
step() {  # step <mode> <config>: hier / predict (TTA) / ensemble / devjoin
  local c; c=$(cfg "$2"); done_zip "$c" && return 0
  py cj.py "$c" "$1"
}
devfold() {  # devfold <base config> <run prefix> <id> <number of dev folds> [key=value ...]: dev cross-fitted members
  local b=$1 p=$2 id=$3 nf=$4 k n; shift 4
  for ((k = 0; k < nf; k++)); do
    n="${p}_df$k"; [ -f "runs/cj/$n/probs.pt" ] && continue
    py cj.py "$(cfg "$b")" train "name=$n" "id=${id}F$k" "cj.devfold=$k" "$@"
    py cj.py "$(cfg "$b")" predict "name=$n" "id=${id}F$k" "cj.devfold=$k" "$@"
  done
}

# Round 1: one technique per architecture, frozen-chunk hierarchical models, ensemble, KD student
for m in c1_gpt_lorma c5_gpt_full c6_gpt_unfreeze c3_long_adalora; do run "$m"; done
py cj.py "$(cfg base)" feats
step hier c8_hier; step hier c12_hier_first; step hier c13_hier_last
for m in c2_hyb_adapter c4_seq_bitfit c7_gpt_partial c14_gpt_lorma_first; do run "$m"; done
step ensemble c9_ens
[ -f runs/cj/teacher.pt ] || py cj.py "$(cfg c10_kd)" teach
run c10_kd
step ensemble c11_ens_final
# Round 2: test-time augmentation over three end windows, hierarchical fine-tuning, ensemble of all (C19)
step predict c17_gpt_lorma_tta; run c15_gpt_lorma_hier
step predict c18_hyb_adapter_tta; run c16_hyb_adapter_hier
step ensemble c19_ens_all
# Round 3: dev cross-fitting (two dev halves, window augmentation + TTA)
A=(cj.augoffsets=[0,1024,2048] cj.tta=[0,1024,2048] cj.trainprobs=false)
devfold c1_gpt_lorma cj_gpt_lorma_dcv C21 2 "${A[@]}";     step devjoin c21_gpt_lorma_dcv
devfold c7_gpt_partial cj_gpt_partial_dcv C22 2 "${A[@]}"; step devjoin c22_gpt_partial_dcv
devfold c2_hyb_adapter cj_hyb_adapter_dcv C20 2 "${A[@]}"; step devjoin c20_hyb_adapter_dcv
devfold c5_gpt_full cj_gpt_full_dcv C23 2 "${A[@]}";       step devjoin c23_gpt_full_dcv
step ensemble c24_ens_dcv
# Round 4: four dev folds, three repeats, length-weighted loss
X=(cj.devfolds=4 cj.devrep=3 cj.lenweight=true cj.augoffsets=[0,1024,2048] cj.tta=[0,1024,2048] cj.trainprobs=false)
devfold c5_gpt_full cj_gpt_full_dcv4 C25 4 "${X[@]}";  step devjoin c25_gpt_full_dcv4
devfold c26_hyb_full cj_hyb_full_dcv4 C26 4 "${X[@]}"; step devjoin c26_hyb_full_dcv4
step ensemble c27_ens_dcv4
say "phase 4 finished; best leaderboard submission: $(zipof "$(cfg c23_gpt_full_dcv)"), train-only system: $(zipof "$(cfg c19_ens_all)")"
