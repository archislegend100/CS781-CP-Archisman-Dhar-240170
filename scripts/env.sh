#!/usr/bin/env bash
# Python environment (once per machine): .venv, the CUDA build of torch matching the driver, requirements,
# the official evaluation scripts, and the CPU self-checks.
# Usage: bash scripts/env.sh [cu118|cu121|cu124|cu126]     PYTHON=python3.11 bash scripts/env.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
$PY -c 'import sys; assert sys.version_info >= (3, 10), "need Python >= 3.10, got " + sys.version'

if [ ! -x .venv/bin/python ]; then
  $PY -m venv .venv 2>/dev/null || { [ -f virtualenv.pyz ] || wget -q https://bootstrap.pypa.io/virtualenv.pyz -O virtualenv.pyz; $PY virtualenv.pyz .venv; }
fi
. .venv/bin/activate

if [ -n "${1:-}" ]; then
  CU=$1
else
  V=$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9]*\.[0-9]*\).*/\1/p' | head -n 1)
  CU=$(awk -v v="$V" 'BEGIN { if (v >= 12.6) print "cu126"; else if (v >= 12.4) print "cu124";
                             else if (v >= 12.1) print "cu121"; else if (v >= 11.8) print "cu118"; else print "none" }')
  echo "driver CUDA $V -> torch wheel $CU"
  [ "$CU" = none ] && { echo "driver CUDA $V is older than 11.8"; exit 1; }
fi
pip install --upgrade pip
pip install torch --index-url "https://download.pytorch.org/whl/$CU"
pip install -r requirements.txt

# Official evaluation scripts (organisers' Google Drive folder, PS section 18.1)
if [ ! -f evaluation_script/evaluate_phase7.py ]; then
  pip install gdown
  gdown --folder "https://drive.google.com/drive/folders/1cLQw89I1KyBotuY3kgjd1nChs-Msksj4" -O evaluation_script \
    || echo "download the evaluation scripts into evaluation_script/ by hand (evaluate_phase1.py ... evaluate_phase7.py)"
fi

export PYTHONPATH="$PWD/src"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not visible to torch'; print('torch', torch.__version__, torch.cuda.get_device_name(0), 'bf16', torch.cuda.is_bf16_supported())"
python src/check.py
