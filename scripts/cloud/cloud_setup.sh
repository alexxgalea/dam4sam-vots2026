#!/bin/bash
# Cloud GPU environment setup for the full-dataset SAM2.1-Large runs.
# Run ON the rented instance (Ubuntu 22.04 + CUDA 12.x image, e.g. RTX 4090 24GB),
# AFTER uploading the code (see docs/CLOUD_SETUP.md).
set -euo pipefail

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$BASE"

echo "== [1/5] uv + Python 3.10 venv (matches local) =="
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
[ -d dam4sam/.venv ] || uv venv dam4sam/.venv --python 3.10
[ -d .venv ] || uv venv .venv --python 3.10

echo "== [2/5] deps (pinned to local versions) =="
uv pip install --python dam4sam/.venv/bin/python \
  torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install --python dam4sam/.venv/bin/python \
  numpy==1.26.4 opencv-python==4.11.0.86 hydra-core==1.3.2 omegaconf==2.3.0 \
  vot-toolkit==0.8.0 scipy==1.15.3 pillow tqdm iopath
# .venv only needs vot for toolkit ops + integrity check
uv pip install --python .venv/bin/python vot-toolkit==0.8.0 numpy==1.26.4

echo "== [3/5] SAM2.1-Large checkpoint =="
mkdir -p dam4sam/checkpoints
[ -f dam4sam/checkpoints/sam2.1_hiera_large.pt ] || \
  wget -q --show-progress -O dam4sam/checkpoints/sam2.1_hiera_large.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

echo "== [4/5] VOTS2026 dataset (~26GB, datacenter-fast) =="
if [ ! -d workspace2026/sequences ]; then
  .venv/bin/vot initialize vots2026/main --workspace workspace2026
fi
echo "sequences: $(ls workspace2026/sequences | wc -l)"

echo "== [5/5] GPU sanity =="
dam4sam/.venv/bin/python - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA not available!"
print(torch.cuda.get_device_name(0),
      f"{torch.cuda.get_device_properties(0).total_memory/1e9:.0f}GB")
EOF

cat <<'EOF'

SETUP DONE. Next (see docs/CLOUD_SETUP.md):
  1. SMOKE TEST (mandatory):  bash scripts/cloud/cloud_run_l.sh smoke
  2. FULL RUN:                bash scripts/cloud/cloud_run_l.sh full
EOF
