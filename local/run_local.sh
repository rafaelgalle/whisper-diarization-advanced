#!/usr/bin/env bash
# Simple helper to create venv, install requirements and run a test transcription.
# Usage: ./run_local.sh /path/to/input.wav
# Must be run from the project root.
set -euo pipefail

INPUT_FILE=${1:-}
if [ -z "$INPUT_FILE" ]; then
  echo "Usage: $0 /path/to/input.wav"
  echo "Example: $0 ./audio.ogg"
  exit 1
fi

echo "Creating and setting up virtual environment..."
python3.11 -m venv .venv
source .venv/bin/activate

echo "Installing project requirements..."
pip install -r ./requirements.txt

echo "Downloading models..."
mkdir -p /models/whisper /models/diarization

# Swap it for large-v3-turbo if you prefer (I got better results with large-v2).
echo "Downloading Whisper model (large-v2)..."
python -c "import faster_whisper; faster_whisper.download_model('large-v2', output_dir='/models/whisper/large-v2')"

echo "Downloading Pyannote diarization model..."
export HF_TOKEN=$(cat /run/secrets/HF_TOKEN 2>/dev/null || echo "hf_TOKEN_HERE")
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='pyannote/speaker-diarization-community-1', local_dir='/models/diarization/pyannote--speaker-diarization-community-1', token='$HF_TOKEN')"

echo "Running transcription on: $INPUT_FILE"
python3 ./main.py --file_path "$INPUT_FILE"
