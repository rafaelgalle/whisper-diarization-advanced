#!/usr/bin/env bash
# Wrapper for cog commands. Syncs core/ from the project root before executing, and cleans up afterwards.
#
# Usage (inside cog/):
#   ./build.sh build -t whisper-diarization-cog
#   ./build.sh predict whisper-diarization-cog:latest -i file_path=@../audio.ogg -i num_speakers=2
#   ./build.sh push r8.im/user/model
#
# The cog build context is the directory where cog.yaml is located (this directory).
# Since core/ is at the project root, it needs to be copied temporarily here
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cleanup() {
  rm -rf "$SCRIPT_DIR/core"
  rm -f "$SCRIPT_DIR/requirements.txt"
}
trap cleanup EXIT

echo "[cog/build.sh] Syncing core/ from root..."
cp -r "$PROJECT_ROOT/core" "$SCRIPT_DIR/core"
cp "$PROJECT_ROOT/core/requirements.txt" "$SCRIPT_DIR/requirements.txt"

# Load HF_TOKEN from common locations
export HF_TOKEN=${HF_TOKEN:-$(cat /run/secrets/HF_TOKEN 2>/dev/null || cat ~/.huggingface/token 2>/dev/null || echo "")}

echo "[cog/build.sh] Executing: cog $*"
cd "$SCRIPT_DIR"

# If we have an HF_TOKEN and build command doesn't already have --secret, add it
if [ -n "$HF_TOKEN" ] && [[ "$*" == *"build"* ]] && [[ "$*" != *"--secret"* ]]; then
  cog "$@" --secret id=HF_TOKEN
else
  cog "$@"
fi
