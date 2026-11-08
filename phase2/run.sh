#!/usr/bin/env bash
# Phase 2 (L-NER): every submission N1-N26 in order on one GPU; the final system is N24.
# Usage: CUDA_VISIBLE_DEVICES=0 bash phase2/run.sh          bash phase2/run.sh one ner.py n2_gpt_lorma train
PHASE=phase2
source "$(dirname "$0")/../scripts/common.sh"
[ "${1:-}" = one ] && { shift; one "$@"; exit 0; }
need $BACKBONES data/phase2/Phase_2_LNER/Train/labels.txt
cfg() { echo "$PHASE/configs/$1.yaml"; }
ep() { python -c "import json, sys; e = [json.load(open(f'runs/ner/{n}/best.json'))['epoch'] for n in sys.argv[1:]]; print(round(sum(e) / len(e)))" "$@"; }

run() {   # single model: train (epoch chosen on 10 held-out docs) + test prediction
  local c; c=$(cfg "$1"); shift; done_zip "$c" && return 0
  py ner.py "$c" train "$@"; py ner.py "$c" predict "$@"
}
fold() {  # cross-validation member or all-docs refit: train only
  local c; c=$(cfg "$1"); shift
  [ -f "runs/ner/$(nameof "$c")/best.pt" ] && { say "skip $(basename "$c" .yaml)"; return 0; }
  py ner.py "$c" train "$@"
}
step() {  # step <mode> <config>: ensemble / oof decisions over trained members
  local c; c=$(cfg "$2"); done_zip "$c" && return 0
  py ner.py "$c" "$1"
}

# Round 1: one technique per architecture, then knowledge distillation from the best token-head run
for m in n1_seq_full n2_gpt_lorma n5_gpt_partial_span n3_hyb_lora n4_long_adapter; do run "$m"; done
run n6_kd "ner.kd.teacher=$(python src/ner.py "$(cfg n6_kd)" best | tail -n 1)"
# Round 2: full fine-tuning, all-docs refits (epochs from the held-out run), second KD student
run n7_gpt_full; run n8_gpt_full_all "ner.epochs=$(ep ner_gpt_full)"
run n9_kd2;      run n10_kd2_all "ner.epochs=$(ep ner_kd2_student)"
step ensemble n11_ens; step ensemble n12_ens_rules
# Round 3: IA3, prompt tuning, LoRMA-pi; 2-fold CV members and their ensembles
for m in n13_gpt_ia3 n14_gpt_prompt n15_gpt_lormapi; do run "$m"; done
for m in n161_lorma_fold1 n162_lorma_fold2 n171_hyb_fold1 n172_hyb_fold2; do fold "$m"; done
step ensemble n16_lorma_cv; step ensemble n17_hyb_cv; step ensemble n18_ens_boundary
# Round 4: out-of-fold calibration (N19) and all-docs refits (N20)
step oof n19_oof
EL=$(ep ner_gpt_lorma_f1 ner_gpt_lorma_f2); EH=$(ep ner_hyb_lora_f1 ner_hyb_lora_f2)
fold n201_lorma_all "ner.epochs=$EL"; fold n203_hyb_all "ner.epochs=$EH"; fold n202_lorma_all_s2 "ner.epochs=$EL"
step oof n20_oof_refit
# Round 5: second-seed and LoRA members, confidence thresholds (N21), wide bag (N22)
fold n223_gptlora_fold1; fold n224_gptlora_fold2
fold n227_gptlora_all "ner.epochs=$(ep ner_gpt_lora_f1 ner_gpt_lora_f2)"
for m in n225_hyb_s2_fold1 n226_hyb_s2_fold2 n221_lorma_s2_fold1 n222_lorma_s2_fold2; do fold "$m"; done
step oof n21_oof_conf; step oof n22_oof_wide
# Round 6: document rules (N23 on N21's bag, N24 on N22's bag = final), then N25/N26 variants
step oof n23_oof_doc; step oof n24_oof_wide_doc
step oof n25_oof_header; step oof n26_oof_diverse
say "phase 2 finished; final submission: $(zipof "$(cfg n24_oof_wide_doc)")"
