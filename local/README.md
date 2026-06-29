# Running Locally

This directory contains everything needed to run Whisper Diarization Advanced on your local machine.

## Quick Start

The `run_local.sh` script handles all setup automatically:
- Creates a Python virtual environment
- Installs dependencies
- Downloads required models (Whisper and Pyannote diarization)
- Runs inference on your audio file

## Prerequisites

- Python 3.11 or later
- Hugging Face access token (for Pyannote model)

## Setup & Running

### Configure your Hugging Face token

You have two options:

#### Option A: Using a secrets file (recommended)

```bash
mkdir -p /run/secrets
echo "hf_your_token_here" > /run/secrets/HF_TOKEN
```

The script will automatically read from this location.

#### Option B: Set environment variable

```bash
export HF_TOKEN="hf_your_token_here"
./run_local.sh audio.wav
```

### Execute the script

```bash
./run_local.sh /path/to/your/audio.wav
```

### Examples

Process a single audio file:
```bash
./run_local.sh ./samples/interview.wav
```

With additional parameters (passed to main.py):
```bash
./run_local.sh ./audio.ogg --num_speakers=2 --preprocess=3 --language=en
```

For a complete list of available parameters and output format, see the [main README](../README.md#input-parameters)
