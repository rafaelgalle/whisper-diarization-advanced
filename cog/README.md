# Running with Cog (Replicate)

This directory contains the Cog configuration for deploying and running Whisper Diarization Advanced on [Replicate.com](https://replicate.com) or locally with Cog.

## What is Cog?

[Cog](https://github.com/replicate/cog) packages machine learning models as containers, making them reproducible and shareable. This setup allows you to:
- Run predictions locally with Cog
- Deploy to Replicate.com with a single command
- Build consistent Docker images for deployment

## Prerequisites

- [Cog](https://github.com/replicate/cog) installed
- Docker installed and running
- Hugging Face access token (for Pyannote model)

## Building & Running

The `build.sh` script is a wrapper that syncs the core logic before executing Cog commands.

### Build the image

```bash
./build.sh build -t whisper-diarization-cog
```

### Run predictions locally

```bash
./build.sh predict whisper-diarization-cog:latest -i file_path=@../audio.ogg -i num_speakers=2
```

### Push to Replicate

```bash
./build.sh push r8.im/your-username/whisper-diarization-advanced
```

## Examples

Basic transcription:
```bash
./build.sh predict whisper-diarization-cog:latest -i file_path=@./audio.wav
```

With preprocessing and language settings:
```bash
./build.sh predict whisper-diarization-cog:latest \
  -i file_path=@./audio.wav \
  -i num_speakers=2 \
  -i preprocess=3 \
  -i language=en
```

With URL input:
```bash
./build.sh predict whisper-diarization-cog:latest \
  -i file_url=https://example.com/audio.wav \
  -i translate=true
```

For a complete list of available parameters and output format, see the [main README](../README.md#input-parameters)
