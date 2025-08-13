### Based on:

This project is based on [thomasmol/whisper-diarization](https://github.com/thomasmol/cog-whisper-diarization), with some enhancements for use with low quality audios and non-english speech.

# Cog Whisper Diarization

Noise reduction + Voice enhancement + High/Low-pass filtering + Audio transcription + Speaker diarization.

Advanced Whisper-based speech diarization pipeline with built-in preprocessing for noisy, multi-channel call center and VoIP audio. Features audio sanitization, noise reduction, channel-based speaker separation, normalization, and future sentiment analysis support. Ideal for low-quality stereo or multi-speaker recordings.

## AI/ML Models used

- Whisper Large v3 ~~Turbo~~ (CTranslate 2 version `faster-whisper==1.1.1`)
- Pyannote audio 3.3.1

## Usage

- try at [Replicate](https://replicate.com/../...)
- Or deploy yourself on [Replicate](https://replicate.com/) or any machine with a GPU 

### Input

- `file_string: str`: Either provide a Base64 encoded audio file.
- `file_url: str`: Or provide a direct audio file URL.
- `file: Path`: Or provide an audio file.
- `num_speakers: int`: Number of speakers. Leave empty to autodetect. Must be between 1 and 50.
- `translate: bool`: Translate the speech into English.
- `language: str`: Language of the spoken words as a language code like 'en'. Leave empty to auto detect language.
- `prompt: str`: Vocabulary: provide names, acronyms, and loanwords in a list. Use punctuation for best accuracy. Also now used as 'hotwords' paramater in transcribing,
- `preprocess: int`: Audio preprocessing level:
  - 0 → No preprocessing (raw audio).
  - 1 → Sanitization only (mono, 16kHz, PCM).
  - 2 → Sanitization + Filtering (highpass & lowpass).
  - 3 → Sanitization + Filtering + Noise reduction.
  - 4 → Sanitization + Filtering + Noise reduction + Normalization.
- `highpass_freq: int`: High-pass filter frequency in Hz (removes low frequencies below this value).
- `lowpass_freq: int`: Low-pass filter frequency in Hz (removes high frequencies above this value).
- `prop_decrease: float`: Noise reduction intensity (0.0 to 1.0), where 1.0 is most aggressive.
- `stationary: bool`: If True, assumes noise is stationary (constant background noise).
- `target_dBFS: float`: Target loudness level for RMS normalization (e.g., -18.0).
  
### Output

- `segments: List[Dict]`: List of segments with speaker, start and end time.
  - Includes `avg_logprob` for each segment and `probability` for each word level segment.
- `num_speakers: int`: Number of speakers (detected, unless specified in input).
- `language: str`: Language of the spoken words as a language code like 'en' (detected, unless specified in input).

### Roadmap / Next Steps
- Sentiment analysis: Classify speech as neutral, negative, or positive.
- Channel-based speaker identification: Detect speakers by audio channel instead of pyannote.
- For stereo call center audio (one channel per speaker), transcribe each channel separately for maximum accuracy.

### Notes & Tips
The higher the noise reduction level, the more vocal characteristics are lost, which can make diarization harder.
(This is why upcoming updates will support channel-based speaker separation.)

Noise reduction is mainly used to improve pause-time detection. Sometimes, background noise can cause incorrect timestamps.

## Thanks to

- [thomasmol/whisper-diarization](https://github.com/thomasmol/cog-whisper-diarization) - Project used as a basis
- [pyannote](https://github.com/pyannote/pyannote-audio) - Speaker diarization model
- [whisper](https://github.com/openai/whisper) - Speech recognition model
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) - Reimplementation of Whisper model for faster inference
- [cog](https://github.com/replicate/cog) - ML containerization framework

