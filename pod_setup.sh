#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# Cloud pod bootstrap for ViZDoom MARL training (RunPod / Vast.ai / Lambda).
# Tested target: Ubuntu 22.04 + NVIDIA CUDA 12.1+ image, root user.
#
# Usage:
#   chmod +x pod_setup.sh
#   ./pod_setup.sh
#
# After it succeeds:
#   source ~/doom_venv/bin/activate
#   tmux new -s doom
#   python doom_deathmatch_ppo.py --save_dir ./runs/run01
#   # detach: Ctrl+B then D     reattach: tmux attach -t doom
# ----------------------------------------------------------------------------
set -euo pipefail

log() { printf '\n\033[1;36m[setup]\033[0m %s\n' "$*"; }
die() { printf '\n\033[1;31m[setup:FAIL]\033[0m %s\n' "$*" >&2; exit 1; }

[ "$(id -u)" -eq 0 ] || SUDO='sudo' && SUDO=${SUDO:-}

# ----------------------------------------------------------------------------
log "Verifying GPU visibility"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi | head -n 20
else
    echo "  WARNING: nvidia-smi not found. Training will fall back to CPU."
fi

# ----------------------------------------------------------------------------
log "Installing system packages (ViZDoom build deps + ffmpeg + tmux)"
export DEBIAN_FRONTEND=noninteractive
$SUDO apt-get update -qq
$SUDO apt-get install -y --no-install-recommends \
    build-essential cmake git wget curl ca-certificates \
    tmux htop nano \
    libboost-all-dev libsdl2-dev libfreetype6-dev libopenal-dev \
    libgme-dev libjpeg-dev libmpg123-dev libsndfile1-dev \
    libwildmidi-dev libgtk2.0-dev nasm libfluidsynth-dev \
    libxmp-dev tar libbz2-dev zlib1g-dev libpng-dev libtiff-dev \
    python3 python3-dev python3-pip python3-venv \
    ffmpeg xvfb pkg-config \
    libgl1-mesa-dri libgl1-mesa-glx libglu1-mesa x11-xserver-utils

# ----------------------------------------------------------------------------
log "Creating Python venv at ~/doom_venv"
python3 -m venv ~/doom_venv
# shellcheck disable=SC1091
source ~/doom_venv/bin/activate
pip install --upgrade pip wheel setuptools

# ----------------------------------------------------------------------------
log "Detecting CUDA version for PyTorch wheel"
CUDA_TAG="cu121"
if command -v nvidia-smi >/dev/null 2>&1; then
    CUDA_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | cut -d. -f1)
    # Rough driver -> CUDA wheel mapping
    if [ -n "${CUDA_MAJOR:-}" ] && [ "$CUDA_MAJOR" -ge 535 ]; then
        CUDA_TAG="cu124"
    elif [ -n "${CUDA_MAJOR:-}" ] && [ "$CUDA_MAJOR" -ge 525 ]; then
        CUDA_TAG="cu121"
    else
        CUDA_TAG="cu118"
    fi
fi
echo "  Using PyTorch wheel index: $CUDA_TAG"

log "Installing PyTorch + numerics"
pip install --no-cache-dir torch torchvision \
    --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

log "Installing ViZDoom + RL deps"
pip install --no-cache-dir \
    "vizdoom>=1.2.3" \
    "opencv-python-headless>=4.10" \
    "numpy<2.1" \
    "imageio>=2.34" \
    "imageio-ffmpeg>=0.5"

# ----------------------------------------------------------------------------
log "Smoke test: CUDA + PyTorch"
python - <<'PY'
import torch
print("torch =", torch.__version__)
print("cuda available =", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device       =", torch.cuda.get_device_name(0))
    print("device count =", torch.cuda.device_count())
PY

log "Smoke test: ViZDoom (headless cig.cfg, under virtual X display)"
# ViZDoom's engine opens a GLX/OpenGL context in init() even with the window
# hidden. On a headless pod there is no X server, so it segfaults. xvfb-run
# provides a virtual display and the Mesa swrast driver renders on CPU.
xvfb-run -a -s '-screen 0 1280x1024x24' python - <<'PY'
import vizdoom as vzd
from pathlib import Path
print("vizdoom =", vzd.__version__)
cfg = Path(vzd.scenarios_path) / "cig.cfg"
assert cfg.is_file(), f"cig.cfg not found at {cfg}"
g = vzd.DoomGame()
g.load_config(str(cfg))
g.set_window_visible(False)
g.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
g.init()
g.new_episode()
for _ in range(8):
    g.make_action([0]*g.get_available_buttons_size(), 1)
print("ViZDoom step OK, screen buf shape =",
      g.get_state().screen_buffer.shape if g.get_state() else None)
g.close()
print("OK")
PY

log "Smoke test: ffmpeg"
ffmpeg -version | head -n 1

# ----------------------------------------------------------------------------
cat <<'EOF'

----------------------------------------------------------------------------
  Setup OK.

  Next steps (note: training MUST run under xvfb-run on a headless pod,
  otherwise the ViZDoom engine segfaults trying to open a GL context):
    source ~/doom_venv/bin/activate
    tmux new -s doom
    xvfb-run -a -s '-screen 0 1280x1024x24' \
        python doom_deathmatch_ppo.py --save_dir ./runs/run01

  In tmux: Ctrl+B then D to detach; `tmux attach -t doom` to reattach.

  Recommended pod sizing:
    - GPU:   RTX 4090 / A10 / A40   (an H100 is wasted here)
    - vCPU:  >= 16 (one ViZDoom process per agent + the learner)
    - RAM:   >= 24 GB
    - Disk:  >= 30 GB (videos + snapshots add up)

  Outputs land in ./runs/run01/:
    policy_latest.pt        -- live weight file workers reload from
    checkpoint_final.pt     -- full PPO state at end of training
    snapshots/              -- league snapshot pool
    videos/                 -- best training episode per agent (640x480)
    eval_videos/            -- pristine 1280x720 final eval recordings
    doom_marl_results.zip   -- bundled deliverable
----------------------------------------------------------------------------
EOF
