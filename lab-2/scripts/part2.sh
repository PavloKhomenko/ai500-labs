#!/usr/bin/env bash
set -euo pipefail

LAB="$(cd "$(dirname "$0")/.." && pwd)"
cd "$LAB"
[ -d /kaggle/working ] && export LAB2_OUTPUT_DIR="${LAB2_OUTPUT_DIR:-/kaggle/working/outputs}"
export TRITON_F32_DEFAULT="${TRITON_F32_DEFAULT:-ieee}"
export TOKENIZERS_PARALLELISM=false
OUT="${LAB2_OUTPUT_DIR:-$LAB/outputs}/part2"

setup() {
  python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
  nvcc --version | tail -2 || true
  nvidia-smi -L
  if ! python -c "import torch, sys; v = tuple(map(int, torch.__version__.split('+')[0].split('.')[:2])); sys.exit(not (2, 9) <= v <= (2, 10))"; then
    pip install -q torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu126
  fi
  pip uninstall -y -q torchaudio || true
  pip install -q -e "$LAB"
  MAMBA_KEEP_CUDA_BUILD=TRUE pip install -q mamba-ssm==2.3.2.post1 --no-build-isolation
  pip install -q causal-conv1d==1.7.0 --no-build-isolation
  python -c "import torch, triton, transformers, peft, mamba_ssm, causal_conv1d; print('torch', torch.__version__, 'triton', triton.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__, 'mamba_ssm', mamba_ssm.__version__, 'causal_conv1d', causal_conv1d.__version__)"
}

train() {
  local configs="${1:-ab}" n_gpu i=0 pids=() status=0
  python -m lab2.part2.train --prepare_only
  n_gpu=$(nvidia-smi -L | wc -l)
  mkdir -p "$OUT"
  if [ "$n_gpu" -ge "${#configs}" ]; then
    for c in $(echo "$configs" | fold -w1); do
      CUDA_VISIBLE_DEVICES=$i python -m lab2.part2.train --cfg "$c" > "$OUT/sft_$c.log" 2>&1 &
      pids+=($!)
      echo "sft_$c on GPU $i, log $OUT/sft_$c.log"
      i=$((i + 1))
    done
    for p in "${pids[@]}"; do wait "$p" || status=1; done
  else
    for c in $(echo "$configs" | fold -w1); do
      CUDA_VISIBLE_DEVICES=0 python -m lab2.part2.train --cfg "$c" 2>&1 | tee "$OUT/sft_$c.log" || status=1
    done
  fi
  for c in $(echo "$configs" | fold -w1); do tail -n 20 "$OUT/sft_$c.log"; done
  return $status
}

case "${1:-}" in
  setup) setup ;;
  eda) python -m lab2.part2.eda ;;
  prepare) python -m lab2.part2.train --prepare_only ;;
  timing) CUDA_VISIBLE_DEVICES="${GPU:-0}" python -m lab2.part2.train --cfg "${2:-a}" --max_steps 20 ;;
  train) train "${2:-ab}" ;;
  verify) CUDA_VISIBLE_DEVICES="${GPU:-0}" python -m lab2.part2.verify "$OUT/sft_${2:-a}" ;;
  *) echo "usage: $0 {setup|eda|prepare|timing [a|b]|train [a|b|ab]|verify [a|b]}"; exit 1 ;;
esac
