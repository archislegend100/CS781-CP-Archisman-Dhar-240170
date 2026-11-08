#!/usr/bin/env bash
# Phase 1 (legal text continuation): every submission in order on one GPU, ending with the final Z13.
# Finished submissions are skipped and interrupted training resumes from last.pt, so the script can be re-run.
# Usage: CUDA_VISIBLE_DEVICES=0 bash phase1/run.sh          bash phase1/run.sh one train.py legalgpt resume
PHASE=phase1
source "$(dirname "$0")/../scripts/common.sh"
[ "${1:-}" = one ] && { shift; one "$@"; exit 0; }
need splits/meta.json data/dev/truth.json
cfg() { echo "$PHASE/configs/$1.yaml"; }

model() {  # train (unless notrain), greedy/sampled test predictions, validated zip
  local c; c=$(cfg "$1"); done_zip "$c" && return 0
  if [ "$(field "$c" "bool(c.get('notrain'))")" != True ]; then
    local run mode=fresh; run=$(field "$c" "c.get('run', 'runs/' + c.name)")
    [ -f "$run/ckpt/last.pt" ] && mode=resume
    py train.py "$c" "$mode"
  fi
  py generate.py "$c" test
  py submit.py "$c"
}
rank() {  # self-ranking: preference pairs (once per pairs file), training, test predictions
  local c; c=$(cfg "$1"); done_zip "$c" && return 0
  [ -f "$(field "$c" "c.rank.get('pairs')")" ] || py selfrank.py "$c" pairs
  py selfrank.py "$c" train
  py generate.py "$c" test
  py submit.py "$c"
}
pool() {  # pool <config> <tag> [key=value ...]: greedy + sampled candidates per test prompt
  local c; c=$(cfg "$1"); local t=$2; shift 2
  [ -f "runs/pool/$t.json" ] || logged "pool_$t" python src/pool.py gen "$c" "tag=$t" "$@"
}
pick() {  # MBR selection over pooled candidates
  local c; c=$(cfg "$1"); done_zip "$c" && return 0
  logged "$(basename "$c" .yaml)" python src/pool.py select "$c"
  py submit.py "$c"
}
X="k=16 nogreedy=true seedoff=100000 temp=0.9"

# Baselines, the four architectures and their ablations (2B tokens each; A* are 0.5B-token ablations)
for m in starter_bpe starter_improved legalgpt gpt_decode legalseq longlegal legalhybrid gpt_rope_s gpt_alibi_s gpt_mha_s hyb_attn0_s; do
  model "$m"
done
# Supreme Court adaptation (+0.4B tokens, 15% replay): S1, S2, P1 (70/30 mix), S3 (second stage)
for m in legalgpt_sc legalhybrid_sc legalhybrid_sc70 legalhybrid_sc2; do model "$m"; done
# Self-ranking post-training: Z1 (ranking loss) and Z3 (DPO) on S1; Z2 and Z4 on S2
rank selfrank; rank z3_selfrank_dpo; rank selfrank_hybrid; rank z4_selfrank_hybrid_dpo
# Decoding-time systems: MBR, ensembles, checkpoint averaging
[ -f runs/legalhybrid_sc/ckpt/avg.pt ] || logged w1_avg python src/avgckpt.py runs/legalhybrid_sc step_500 best last
for m in z5_mbr m1_mbr e1_ens m2_ensmbr z6_ensmbr w1_avg; do model "$m"; done
# Candidate pools and MBR selection: F1-F3 (Z3 + S2), Z7 (Z3 + S3)
pool z5_mbr z3 k=15; pool legalhybrid_sc s2 k=15; pool legalhybrid_sc2 s3 k=15
for m in f1_lex f2_emb f3_mix z7_pool; do pick "$m"; done
# 64-candidate pools: Z8-Z10; Z11 = embedding-reward DPO; Z12; Z13 (final: 3-layer embedding utility)
pool z5_mbr z3b $X; pool legalhybrid_sc2 s3b $X
for m in z8_pool z9_qmbr z10_mmu; do pick "$m"; done
c=$(cfg selfrank_emb)
if ! done_zip "$c"; then
  py selfrank.py "$c" rerank; py selfrank.py "$c" train; py generate.py "$c" test; py submit.py "$c"
fi
pool selfrank_emb ze k=15; pool selfrank_emb zeb $X
pick z12_pool
pick z13_final
say "phase 1 finished; final submission: $(zipof "$(cfg z13_final)")"
