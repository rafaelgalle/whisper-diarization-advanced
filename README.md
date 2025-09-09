# Whisper Diarization Advanced

**Ultra-fast, customizable speech-to-text and speaker diarization for noisy, multi-speaker audio. Includes advanced noise reduction, stereo channel support, and flexible audio preprocessing—ideal for call centers, meetings, and podcasts**

---

<img width="2842" height="1553" alt="image" src="https://github.com/user-attachments/assets/663f4523-7f0f-4662-8353-fe4b838618e3" />


## Why Use This Project?

- **Ultra Fast & Cost-Effective:** Optimized for Replicate.com and GPU/CPU environments, delivering rapid results at minimal cost.
- **Highly Customizable:** Choose your model, device, and audio preprocessing level. Fine-tune for your use case—call centers, interviews, podcasts, meetings, and more.
- **Advanced Audio Treatment:** Built-in options for sanitization, high/low-pass filtering, aggressive noise reduction, and RMS normalization. Tame even the worst audio!
- **Stereo Channel Support:** Perfect for call center recordings—transcribes each channel separately for maximum speaker accuracy.
- **Multi-Input Flexibility:** Accepts direct file upload, URL, or base64 string. Integrate easily with any workflow.
- **Speaker Diarization & Transcription:** State-of-the-art Whisper and Pyannote models for accurate speech-to-text and speaker separation.
- **Translation & Language Detection:** Auto-detects language and can translate speech to English for global applications.
- **Scalable & Production-Ready:** Designed for batch processing, API integration, and large-scale deployments.

---

## Features

- **Noise Reduction + Voice Enhancement**
- **High/Low-Pass Filtering**
- **Audio Sanitization (mono, 16kHz, PCM)**
- **Channel-Based Speaker Separation**
- **RMS Normalization**
- **Sentiment Analysis (roadmap)**
- **Custom Vocabulary/Hotwords**
- **Flexible Preprocessing Levels (0-4)**

---

## Usage

Try it instantly on [Replicate.com](https://replicate.com/rafaelgalle/whisper-diarization-advanced) or deploy on your own GPU/CPU server.

### Running Locally with Cog

1. Install and configure [Cog](https://replicate.com/docs/cog):
   ```bash
   pip install cog
   cog init  # if you haven't initialized a Cog project yet
2. Once your model is set up, you can run predictions locally:
   ```bash
   cog predict -i file_path=@input.wav -i num_speakers=2
3. You can pass any other parameters supported by the model in the same way
   ```bash
   cog predict -i file_path=@input.wav -i num_speakers=2 -i some_param=value
### Input Options

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


### Notes & Tips
The higher the noise reduction level, the more vocal characteristics are lost, which can make diarization harder.
(This is why upcoming updates will support channel-based speaker separation.)

Noise reduction is mainly used to improve pause-time detection. Sometimes, background noise can cause incorrect timestamps.

---

## Ideal Use Cases

- **Call Centers:** Stereo channel separation for agent/customer
- **Meetings & Interviews:** Multi-speaker diarization
- **Podcasts & Broadcasts:** Clean transcription from noisy sources
- **VoIP & Low-Quality Audio:** Advanced noise handling

---

## Roadmap / Next Steps
- Sentiment analysis: Classify speech as neutral, negative, or positive
- Summarize content
- PII Mask

## Cost
- **$0.000975 per second of execution**

## Speed & Cost
- **Mono**  
  Using an *Nvidia L40S GPU*, transcribes **5 minutes** of audio (2 speakers) in **8.1 seconds** — cost: **$0.0079**

- **Stereo**  
  Using an *Nvidia L40S GPU*, transcribes **5 minutes** of audio (2 speakers) in **4.6 seconds** — cost: **$0.0044**

## Thanks to

- [thomasmol/whisper-diarization](https://github.com/thomasmol/cog-whisper-diarization)
- [pyannote](https://github.com/pyannote/pyannote-audio)
- [whisper](https://github.com/openai/whisper)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [cog](https://github.com/replicate/cog)

---

**Ready to get started? Try it now on Replicate.com and experience the fastest, most customizable speech diarization for your audio!**

