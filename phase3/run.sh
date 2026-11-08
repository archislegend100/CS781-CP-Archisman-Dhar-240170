#!/usr/bin/env bash
# Phase 3 (rhetorical roles): every submission R1-R31 in order on one GPU; the final system is R30.
# Usage: CUDA_VISIBLE_DEVICES=0 bash phase3/run.sh          bash phase3/run.sh one rr.py r1_gpt_lorma train
PHASE=phase3
source "$(dirname "$0")/../scripts/common.sh"
[ "${1:-}" = one ] && { shift; one "$@"; exit 0; }
need $BACKBONES runs/selfrank_dpo/ckpt/best.pt
cfg() { echo "$PHASE/configs/$1.yaml"; }
epoch() { python -c "import json, sys; print(json.load(open(sys.argv[1]))['epoch'])" "runs/rr/$1/best.json"; }

run() {   # fine-tuned context model: train (epoch chosen on dev) + emissions for train/dev/test + submission
  local c; c=$(cfg "$1"); shift; done_zip "$c" && return 0
  py rr.py "$c" train "$@"; py rr.py "$c" predict "$@"
}
step() {  # step <mode> <config> [key=value ...]: seq / cv / ensemble / cvjoin / cvens
  local m=$1 c; c=$(cfg "$2"); shift 2; done_zip "$c" && return 0
  py rr.py "$c" "$m" "$@"
}
crossfit() {  # crossfit <base config> <run prefix> <id> <epochs>: 5 fold models over train + dev
  local k n
  for k in 0 1 2 3 4; do
    n="$2_f$k"; [ -f "runs/rr/$n/emissions.pt" ] && continue
    py rr.py "$(cfg "$1")" train "name=$n" "id=$3F$k" "rr.foldk=$k" "rr.epochs=$4"
    py rr.py "$(cfg "$1")" predict "name=$n" "id=$3F$k" "rr.foldk=$k"
  done
}

# Round 1: one technique per architecture (hierarchical context model), ensemble, KD student
for m in r1_gpt_lorma r5_gpt_full r3_long_qlora r11_gpt_prompt r2_hyb_rslora r4_seq_adalora r6_gpt_ia3 r7_gpt_lorma_sent; do run "$m"; done
step ensemble r8_ens
[ -f runs/rr/teacher.pt ] || py rr.py "$(cfg r9_kd)" teach
run r9_kd
# R10: the three best single context models refitted on train + dev for their best epoch count
RUNS=runs/rr/rr_kd_student
for t in $(python - <<'PY'
import glob, json
rows = []
for f in glob.glob("runs/rr/*/best.json"):
    b = json.load(open(f))
    if b.get("dev") and b.get("mode", "ctx") == "ctx" and b.get("method") != "kd" and not b["name"].endswith("_all"):
        rows.append((b["dev"]["f1"], b["name"], b["id"], b["epoch"]))
print(" ".join(f"{n}:{i}:{e}" for _, n, i, e in sorted(rows, reverse=True)[:3]))
PY
); do
  IFS=: read -r n i e <<< "$t"
  c=$(grep -l "^name: $n\$" $PHASE/configs/*.yaml | head -n 1)
  if [ ! -f "runs/rr/${n}_all/emissions.pt" ]; then
    py rr.py "$c" train rr.alldocs=true "name=${n}_all" "id=${i}A" "rr.epochs=$e"
    py rr.py "$c" predict rr.alldocs=true "name=${n}_all" "id=${i}A"
  fi
  RUNS="$RUNS,runs/rr/${n}_all"
done
step ensemble r10_ens_refit "ens.runs=[$RUNS]"
# Round 2: frozen whole-document features of seven Phase-1 checkpoints -> BiLSTM-CRF document taggers
[ -f runs/rr/feats/z3.pt ] || py rr.py "$(cfg feats)" feats
step seq r12_seq4; step seq r13_seq_wide
step ensemble r14_ens; step ensemble r15_ens4
step seq r16_seq_wide_all "seq.epochs=$(epoch rr_seq_wide)"; step ensemble r17_ens_all
# Round 3: cross-fitted document taggers (5 folds x 3 seeds over train + dev, out-of-fold calibration)
for m in r18_cv_lex r19_cv_lex_dom r20_cv_nolex; do step cv "$m"; done
step ensemble r21_ens_cv
step cv r22_cv_pca; step cv r23_cv_pca_small; step ensemble r24_ens_pca
step cv r25_cv_sparse; step cv r27_cv_sparse_reg; step ensemble r26_ens_sparse
# Round 4: cross-fitted fine-tuned context models (R28 hybrid rsLoRA, R29 LoRMA+) and the final ensemble R30
crossfit r2_hyb_rslora rr_hyb_cf R28 4;      step cvjoin r28_cf_hyb
crossfit r1_gpt_lorma rr_gpt_lorma_cf R29 8; step cvjoin r29_cf_lorma
step cvens r30_cvens; step cvens r31_cvens_dom
say "phase 3 finished; final submission: $(zipof "$(cfg r30_cvens)")"
