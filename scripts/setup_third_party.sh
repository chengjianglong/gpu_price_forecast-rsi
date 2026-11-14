#!/usr/bin/env bash
# Clone the benchmark and the two world models at pinned commits into third_party/.
# Model weights are large and are not downloaded here; see README.md.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/third_party"
mkdir -p "$DEST"

clone() {  # name url commit [sparse-path]
  local name="$1" url="$2" commit="$3" sparse="${4:-}"
  local dir="${DEST}/${name}"
  if [[ -d "${dir}/.git" ]]; then
    echo "== ${name}: already present"
  else
    echo "== ${name}: cloning"
    git clone --filter=blob:none --no-checkout "$url" "$dir"
    if [[ -n "$sparse" ]]; then
      git -C "$dir" sparse-checkout set "$sparse"
    fi
  fi
  git -C "$dir" fetch --quiet origin "$commit" || true
  git -C "$dir" checkout --quiet "$commit"
  echo "   ${name} @ $(git -C "$dir" rev-parse --short HEAD)"
}

clone PlayWorld   https://github.com/kxding/PlayWorld.git          eeaea12e050a90b815d11d030e943e7b3181da21
clone ABot-World  https://github.com/amap-cvlab/ABot-World.git     5c3f7260f2346ebb60a49efb81a7db5bfe15d7e5
# Only Matrix-Game-3/ is needed from the multi-version repo.
clone Matrix-Game https://github.com/SkyworkAI/Matrix-Game.git     51913b2bfa78a6d54de912b8348b1d318f4d36e2 Matrix-Game-3

cat <<'EOF'

Next steps (on the GPU machine):
  ABot-World-0 weights:
    hf download acvlab/ABot-World-0-5B-LF --local-dir third_party/ABot-World/checkpoints/ABot-World-0-5B-LF
  Matrix-Game 3.0 weights:
    hf download Skywork/Matrix-Game-3.0 --local-dir third_party/Matrix-Game/Matrix-Game-3/Matrix-Game-3.0
  Each model has its own environment (PyTorch, FlashAttention, ...): follow its README.
  PlayWorld scorer: pip install -r third_party/PlayWorld/requirements.txt
EOF
