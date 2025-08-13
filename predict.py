import base64
import subprocess
import os
import requests
import time
import torch
import re
import pandas as pd
import numpy as np
from cog import BasePredictor, BaseModel, Input, Path
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
import torchaudio
from faster_whisper.vad import VadOptions
from scipy.io import wavfile
import noisereduce as nr
from scipy.io.wavfile import write

def sanitize_audio(input_path, sanitized_path):
    """Convert audio to mono, 16kHz, pcm_s16le using ffmpeg."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        sanitized_path
    ]
    subprocess.run(cmd, check=True)
    print(f"Sanitization completed: {sanitized_path}")

def apply_filters(input_path, filtered_path, highpass_freq, lowpass_freq):
    """Applies highpass and lowpass filters via ffmpeg."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-af", f"highpass=f={highpass_freq},lowpass=f={lowpass_freq}",
        filtered_path
    ]
    subprocess.run(cmd, check=True)
    print(f"Applied filters: {filtered_path}")

def reduce_noise(input_path, prop_decrease, stationary):
    """Reduces audio noise and returns numpy float32 data and sample rate."""
    rate, data = wavfile.read(input_path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    reduced = nr.reduce_noise(
        y=data,
        sr=rate,
        prop_decrease=prop_decrease,
        stationary=stationary,
    )
    print("Noise removed")
    return rate, reduced

def normalize_audio(audio_data, target_dBFS):
    """Normalizes audio RMS to target dBFS."""
    rms = np.sqrt(np.mean(audio_data**2))
    target_rms = 10 ** (target_dBFS / 20)
    gain = target_rms / (rms + 1e-9)
    print("Audio normalized")
    return audio_data * gain

def preprocess_audio(input_path, output_path,
                     preprocess_level: int = 4,
                     highpass_freq=45, lowpass_freq=8000,
                     prop_decrease=1.0, stationary=False,
                     target_dBFS=-20.0):
    """
    Performs audio cleanup to the specified level.

    Levels:
    0 - Does nothing (copies the original file)
    1 - Sanitization
    2 - Sanitization + Filter
    3 - Sanitization + Filter + ReduceNoise
    4 - Sanitization + Filter + ReduceNoise + Normalization
    """
    temp1 = "temp_stage1.wav"
    temp2 = "temp_stage2.wav"
    temp3 = "temp_stage3.wav"

    if preprocess_level == 0:
        subprocess.run(["cp", input_path, output_path], check=True)
        return

    sanitize_audio(input_path, temp1)
    if preprocess_level == 1:
        subprocess.run(["cp", temp1, output_path], check=True)
        return

    apply_filters(temp1, temp2, highpass_freq, lowpass_freq)
    if preprocess_level == 2:
        subprocess.run(["cp", temp2, output_path], check=True)
        return

    rate, reduced = reduce_noise(temp2, prop_decrease, stationary)
    out_int16 = np.clip(reduced * 32768, -32768, 32767).astype(np.int16)
    write(temp3, rate, out_int16)
    if preprocess_level == 3:
        subprocess.run(["cp", temp3, output_path], check=True)
        return

    normalized = normalize_audio(reduced, target_dBFS)
    out_int16 = np.clip(normalized * 32768, -32768, 32767).astype(np.int16)
    write(output_path, rate, out_int16)

    print(f"Preprocessing complete. Temp file saved in: {output_path}")

class Output(BaseModel):
    segments: list
    language: str = None
    num_speakers: int = None

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        #model_name = "large-v3-turbo"
        model_name = "large-v3"
        self.model = WhisperModel(
            model_name,
            device="cuda",
            compute_type="float16",
        )
        self.diarization_model = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token="YOUR HF TOKEN",
        ).to(torch.device("cuda"))

    def predict(
        self,
        file_string: str = Input(
            description="Either provide: Base64 encoded audio file,", default=None
        ),
        file_url: str = Input(
            description="Or provide: A direct audio file URL", default=None
        ),
        file_path: Path = Input(description="Or an audio file", default=None),
        num_speakers: int = Input(
            description="Number of speakers, leave empty to autodetect.",
            ge=1,
            le=50,
            default=None,
        ),
        translate: bool = Input(
            description="Translate the speech into English.",
            default=False,
        ),
        language: str = Input(
            description="Language of the spoken words as a language code like 'en', 'pt'... Leave empty to auto detect language.",
            default=None,
        ),
        prompt: str = Input(
            description="Vocabulary: provide names, acronyms and loanwords in a list. Use punctuation for best accuracy.",
            default=None,
        ),
        preprocess: int = Input(
            description="Preprocess levels: 0 = None, 1 = Sanitization, 2 = +Filter, 3 = +ReduceNoise, 4 = +Normalization.",
            ge=0,
            le=4,
            default=1,
        ),
        highpass_freq: int = Input(
            description=(
                "High-pass filter cutoff frequency in Hz. Removes low-frequency noise below this value "
                "(e.g., air conditioning hum, mic handling noise). Default is 45."
            ),
            default=45
        ),
        lowpass_freq: int = Input(
            description=(
                "Low-pass filter cutoff frequency in Hz. Removes high-frequency noise above this value "
                "(e.g., dog barks, keyboard clicks, hiss). Default is 8000."
            ),
            default=8000
        ),
        prop_decrease: float = Input(
            description=(
                "Noise reduction intensity (0.0 to 1.0). Higher values remove more noise "
                "(e.g., fan noise, static) but may also remove voice characteristics, making diarization harder. "
                "Default is 0.3."
            ),
            default=0.3
        ),
        stationary: bool = Input(
            description=(
                "If True, assumes the noise profile is constant throughout the audio "
                "(best for steady background hum). Default is True."
            ),
            default=True
        ),
        target_dBFS: float = Input(
            description=(
                "Target volume level in decibels for normalization. "
                "For example, -18.0 boosts quiet voices or reduces overly loud parts for consistent playback. "
                "Default is -18.0."
            ),
            default=-18.0
        ),
    ) -> Output:
        """Run a single prediction on the model"""

        temp_input = f"temp_input_{time.time_ns()}.wav"
        temp_processed = f"temp_processed_{time.time_ns()}.wav"

        try:
            if file_path:
                subprocess.run(["ffmpeg", "-y", "-i", file_path, temp_input], check=True)
            elif file_url:
                r = requests.get(file_url)
                with open(temp_input, "wb") as f:
                    f.write(r.content)
            elif file_string:
                audio_data = base64.b64decode(file_string.split(",")[1] if "," in file_string else file_string)
                with open(temp_input, "wb") as f:
                    f.write(audio_data)

            if preprocess > 0:
                preprocess_audio(
                    temp_input,
                    temp_processed,
                    preprocess_level=preprocess,
                    highpass_freq=highpass_freq,
                    lowpass_freq=lowpass_freq,
                    prop_decrease=prop_decrease,
                    stationary=stationary,
                    target_dBFS=target_dBFS
                )
                audio_for_model = temp_processed
            else:
                audio_for_model = temp_input

            segments, detected_num_speakers, detected_language = self.speech_to_text(
                audio_for_model, num_speakers, prompt, language, translate
            )

            print(f"Done")
            return Output(segments, detected_language, detected_num_speakers)

        except Exception as e:
            raise RuntimeError("Error Running inference with local model", e)
        
        finally:
            for f in [temp_input, temp_processed]:
                if os.path.exists(f):
                    os.remove(f)

    def speech_to_text(
        self,
        audio_file_wav,
        num_speakers=None,
        prompt="",
        language=None,
        translate=False,
    ):
        time_start = time.time()

        print("Starting transcribing")
        options = dict(
            language=language,
            beam_size=5,
            vad_filter=True,
            vad_parameters=VadOptions(
                max_speech_duration_s=self.model.feature_extractor.chunk_length,
                min_speech_duration_ms=100,
                speech_pad_ms=100,
                threshold=0.25,
                neg_threshold=0.2,
            ),
            word_timestamps=True,
            initial_prompt=prompt,
            language_detection_segments=1,
            task="translate" if translate else "transcribe",
        )
        segments, transcript_info = self.model.transcribe(audio_file_wav, **options)
        segments = list(segments)
        segments = [
            {
                "avg_logprob": s.avg_logprob,
                "start": float(s.start),
                "end": float(s.end),
                "text": s.text,
                "words": [
                    {
                        "start": float(w.start),
                        "end": float(w.end),
                        "word": w.word,
                        "probability": w.probability,
                    }
                    for w in s.words
                ],
            }
            for s in segments
        ]

        time_transcribing_end = time.time()
        print(
            f"Finished with transcribing, took {time_transcribing_end - time_start:.5} seconds, {len(segments)} segments"
        )

        print("Starting diarization")
        waveform, sample_rate = torchaudio.load(audio_file_wav)
        diarization = self.diarization_model(
            {"waveform": waveform, "sample_rate": sample_rate},
            num_speakers=num_speakers,
        )

        time_diraization_end = time.time()
        print(
            f"Finished with diarization, took {time_diraization_end - time_transcribing_end:.5} seconds"
        )

        print("Starting merging")

        # Convert diarization list to DataFrame
        diarize_segments = []
        diarization_list = list(diarization.itertracks(yield_label=True))

        for turn, _, speaker in diarization_list:
            diarize_segments.append(
                {"start": turn.start, "end": turn.end, "speaker": speaker}
            )
        diarize_df = pd.DataFrame(diarize_segments)
        unique_speakers = {speaker for _, _, speaker in diarization_list}
        detected_num_speakers = len(unique_speakers)

        # Process each segment and its words
        final_segments = []
        for segment in segments:
            # Calculate intersection for segment-level speaker assignment
            diarize_df["intersection"] = np.minimum(
                diarize_df["end"], segment["end"]
            ) - np.maximum(diarize_df["start"], segment["start"])
            diarize_df["union"] = np.maximum(
                diarize_df["end"], segment["end"]
            ) - np.minimum(diarize_df["start"], segment["start"])

            # Get speaker with maximum intersection
            dia_tmp = diarize_df[diarize_df["intersection"] > 0]
            if len(dia_tmp) > 0:
                speaker = (
                    dia_tmp.groupby("speaker")["intersection"]
                    .sum()
                    .sort_values(ascending=False)
                    .index[0]
                )
            else:
                speaker = "UNKNOWN"

            # Process words if they exist
            words_with_speakers = []
            for word in segment["words"]:
                # Calculate intersection for word-level speaker assignment
                diarize_df["intersection"] = np.minimum(
                    diarize_df["end"], word["end"]
                ) - np.maximum(diarize_df["start"], word["start"])
                diarize_df["union"] = np.maximum(
                    diarize_df["end"], word["end"]
                ) - np.minimum(diarize_df["start"], word["start"])

                # Get speaker with maximum intersection
                dia_tmp = diarize_df[diarize_df["intersection"] > 0]
                if len(dia_tmp) > 0:
                    word_speaker = (
                        dia_tmp.groupby("speaker")["intersection"]
                        .sum()
                        .sort_values(ascending=False)
                        .index[0]
                    )
                else:
                    word_speaker = (
                        speaker  # Fall back to segment speaker if no intersection
                    )

                word["speaker"] = word_speaker
                words_with_speakers.append(word)

            # Create new segment with speaker information
            new_segment = {
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"],
                "speaker": speaker,
                "avg_logprob": segment["avg_logprob"],
                "words": words_with_speakers,
            }
            final_segments.append(new_segment)

        # Smart grouping of segments
        if len(final_segments) > 0:
            grouped_segments = []
            current_group = final_segments[0].copy()
            sentence_end_pattern = r"[.!?]+"

            for segment in final_segments[1:]:
                time_gap = segment["start"] - current_group["end"]
                current_duration = current_group["end"] - current_group["start"]

                # Conditions for combining segments:
                # 1. Same speaker
                # 2. Time gap is reasonable (≤ 1 second)
                # 3. Current group doesn't end with sentence-ending punctuation
                # 4. Combined duration would not exceed 30 seconds
                can_combine = (
                    segment["speaker"] == current_group["speaker"]
                    and time_gap <= 1.0
                    and current_duration < 30.0
                    and not re.search(sentence_end_pattern, current_group["text"][-1:])
                )

                if can_combine:
                    # Merge segments
                    current_group["end"] = segment["end"]
                    current_group["text"] += " " + segment["text"]
                    # current_group["words"].extend(segment["words"])
                else:
                    # Start new group
                    grouped_segments.append(current_group)
                    current_group = segment.copy()

            grouped_segments.append(current_group)
            final_segments = grouped_segments

        # Final cleanup of text
        for segment in final_segments:
            # Remove extra spaces
            segment["text"] = re.sub(r"\s+", " ", segment["text"]).strip()
            # Ensure proper spacing around punctuation
            segment["text"] = re.sub(r"\s+([.,!?])", r"\1", segment["text"])
            # Calculate segment duration
            segment["duration"] = segment["end"] - segment["start"]

        time_merging_end = time.time()
        print(
            f"Finished with merging, took {time_merging_end - time_diraization_end:.5} seconds"
        )

        return final_segments, detected_num_speakers, transcript_info.language
