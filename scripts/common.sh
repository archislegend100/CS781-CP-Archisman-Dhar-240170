# Shared helpers, sourced by phase*/setup.sh and phase*/run.sh. Every path is relative to the repository root.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
[ -f .venv/bin/activate ] && source .venv/bin/activate
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p logs

say() { echo "== $(date '+%F %T') $*"; }

# field <config> <python expression over c>
field() { python -c "import sys; from config import loadcfg; c = loadcfg(sys.argv[1]); print($2)" "$1"; }
zipof() { field "$1" "'submissions/%s_%s/submission.zip' % (c.id, c.get('subname', c.name))"; }
nameof() { field "$1" "c.name"; }

# logged <log stem> <command ...>: append output to logs/<stem>.log; stop the pipeline on failure
# DRYRUN=1 prints the commands without running them.
logged() {
  local log="logs/$1.log"; shift
  say "$*"
  [ -n "${DRYRUN:-}" ] && return 0
  "$@" >> "$log" 2>&1 || { say "FAILED: $* (see $log)"; exit 1; }
}
# py <script> <config> [mode ...] [key=value ...]
py() { local s=$1 c=$2; shift 2; logged "$(basename "$c" .yaml)" python "src/$s" "$c" "$@"; }

# done_zip <config>: true when the config's submission.zip already exists (finished steps are skipped on re-runs)
done_zip() { local z; z=$(zipof "$1"); [ -f "$z" ] && say "skip $(basename "$1" .yaml) ($z exists)"; }

need() { [ -n "${DRYRUN:-}" ] && return 0; for f in "$@"; do [ -e "$f" ] || { say "missing $f: run the earlier phase or this phase's setup.sh first"; exit 1; }; done; }

# Phase-1 backbones used by Phases 2-7 (trained by phase1/run.sh)
BACKBONES="runs/legalgpt/ckpt/best.pt runs/legalgpt_sc/ckpt/last.pt runs/legalhybrid/ckpt/best.pt
  runs/legalhybrid_sc2/ckpt/last.pt runs/longlegal/ckpt/best.pt runs/legalseq/ckpt/best.pt"

# one <script> <config> <mode> [key=value ...]: run any single step by hand, e.g. bash phase2/run.sh one ner.py n2_gpt_lorma train
one() { local s=$1 c=$2; shift 2; [ -f "$c" ] || c="$PHASE/configs/$c.yaml"; py "$s" "$c" "$@"; }
