"""Speech-to-text + diarization pipeline (shared by local & Cog)."""
from __future__ import annotations
import base64
import subprocess
import os
import requests
import logging
import torch
import torchaudio
import tempfile
import shutil
import re
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional, Tuple
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline as PyannotePipeline
from faster_whisper.vad import VadOptions
from preprocess import preprocess_audio
from metrics_utils import AudioMetrics, SpeechActivityMetricsCalculator

log = logging.getLogger(__name__)


def _collect_cuda_diagnostics() -> Dict[str, Optional[str]]:
    diagnostics: Dict[str, Optional[str]] = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": None,
        "cuda_device_name": None,
        "ld_library_path": os.getenv("LD_LIBRARY_PATH"),
    }

    try:
        diagnostics["cuda_available"] = str(torch.cuda.is_available())
    except Exception as exc:
        diagnostics["cuda_available"] = f"error: {exc}"
        return diagnostics

    if diagnostics["cuda_available"] == "True":
        try:
            diagnostics["cuda_device_name"] = torch.cuda.get_device_name(0)
        except Exception as exc:
            diagnostics["cuda_device_name"] = f"error: {exc}"

    return diagnostics

class Output:
    def __init__(
        self,
        segments: List[Dict],
        metrics: AudioMetrics,
        language: Optional[str] = None,
        num_speakers: Optional[int] = None,
    ):
        self.segments = segments
        self.language = language
        self.num_speakers = num_speakers
        self.metrics = metrics

    def to_dict(self) -> Dict:
        return {
            "segments": self.segments,
            "language": self.language,
            "num_speakers": self.num_speakers,
            "metrics": self.metrics.to_dict(),
        }

WHISPER_MODEL_PATH = "/models/whisper/large-v2"
DIARIZATION_MODEL_PATH = "/models/diarization/pyannote--speaker-diarization-community-1"

class WhisperDiarizationPipeline:
    def __init__(self, device: str = "cpu", compute_type: str = "int8"):
        """Load models into memory."""

        diagnostics = _collect_cuda_diagnostics()
        log.info(
            "Pipeline bootstrap starting: device=%s compute_type=%s torch=%s torch_cuda=%s cuda_available=%s cuda_device=%s ld_library_path=%s",
            device,
            compute_type,
            diagnostics["torch_version"],
            diagnostics["torch_cuda_version"],
            diagnostics["cuda_available"],
            diagnostics["cuda_device_name"],
            diagnostics["ld_library_path"],
        )

        log.info("Whisper model path: %s | Diarization model path: %s", WHISPER_MODEL_PATH, DIARIZATION_MODEL_PATH)

        if device.startswith("cuda") and diagnostics["cuda_available"] != "True":
            raise RuntimeError(
                "CUDA device requested but torch.cuda.is_available() is not True. "
                f"torch={diagnostics['torch_version']} torch.version.cuda={diagnostics['torch_cuda_version']} "
                f"cuda_available={diagnostics['cuda_available']} ld_library_path={diagnostics['ld_library_path']}"
            )

        log.info("Loading faster-whisper model into memory")
        self.model = WhisperModel(
            WHISPER_MODEL_PATH,
            device=device,
            compute_type=compute_type,
            local_files_only=True,
        )
        log.info("Loading pyannote diarization pipeline into memory")
        self.diarization_model = PyannotePipeline.from_pretrained(
            DIARIZATION_MODEL_PATH,
        ).to(torch.device(device))
        log.info("Pipeline bootstrap finished successfully")

    def _get_file(self, file_path=None, file_url=None, file_string=None) -> str:
        """
        Handles any input type and converts it to PCM 16kHz WAV.
        Returns the path to the converted file.
        """

        if not any([file_path, file_url, file_string]):
            raise ValueError("One of file_path, file_url, or file_string must be provided.")

        temp_dir = tempfile.mkdtemp()
        raw_path = os.path.join(temp_dir, "input_raw")
        processed_path = os.path.join(temp_dir, "input_pcm16.wav")

        # Save raw input
        if file_path:
            shutil.copy(file_path, raw_path)
        elif file_url:
            r = requests.get(file_url, timeout=60)
            r.raise_for_status()
            with open(raw_path, "wb") as f:
                f.write(r.content)
        elif file_string:
            audio_bytes = base64.b64decode(file_string.split(",")[1] if "," in file_string else file_string)
            with open(raw_path, "wb") as f:
                f.write(audio_bytes)

        # Convert to PCM 16kHz
        subprocess.run([
            "ffmpeg", "-y", "-i", raw_path,
            "-acodec", "pcm_s16le", "-ar", "16000",
            # "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000",
            processed_path
        ], check=True)

        return processed_path, temp_dir

    def predict(
        self,
        file_string: Optional[str] = None,
        file_url: Optional[str] = None,
        file_path: Optional[str] = None,
        num_speakers: Optional[int] = None,
        translate: bool = False,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        preprocess: int = 4,
        highpass_freq: int = 45,
        lowpass_freq: int = 8000,
        prop_decrease: float = 1.0,
        stationary: bool = True,
        target_dBFS: float = -18.0
    ) -> Output:
        """Run a single prediction on the model."""
        temp_input, temp_dir = self._get_file(file_path, file_url, file_string)

        try:
            num_channels = self._get_audio_channels(temp_input)
            print(f"DEBUG --> Audio with {num_channels} channels")
            if num_channels == 1:
                temp_processed = os.path.join(temp_dir, "input_processed.wav")
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
            
                print(f"DEBUG --> Starting transcribing mono")
                segments, detected_num_speakers, detected_language = self.speech_to_text(
                    audio_for_model, num_speakers, prompt or "", language, translate
                )
                metrics = self._build_metrics(audio_for_model, segments, num_channels=num_channels)
                return Output(segments, metrics, language=detected_language, num_speakers=detected_num_speakers)

            else:
                print(f"DEBUG --> Spliting channels")
                ch1_path, ch2_path = self._split_stereo_channels(temp_input, temp_dir)
                ch1_proc = os.path.join(temp_dir, "ch1_proc.wav")
                ch2_proc = os.path.join(temp_dir, "ch2_proc.wav")

                if preprocess > 0:
                    preprocess_audio(ch1_path, ch1_proc,
                                    preprocess_level=preprocess,
                                    highpass_freq=highpass_freq,
                                    lowpass_freq=lowpass_freq,
                                    prop_decrease=prop_decrease,
                                    stationary=stationary,
                                    target_dBFS=target_dBFS)
                    preprocess_audio(ch2_path, ch2_proc,
                                    preprocess_level=preprocess,
                                    highpass_freq=highpass_freq,
                                    lowpass_freq=lowpass_freq,
                                    prop_decrease=prop_decrease,
                                    stationary=stationary,
                                    target_dBFS=target_dBFS)
                else:
                    ch1_proc = ch1_path
                    ch2_proc = ch2_path
                
                def _run_transcription(channel_id: int, audio_path: str):
                    print(f"DEBUG --> Starting transcribing stereo channel {channel_id}")
                    return channel_id, *self._transcribe_audio(audio_path, language, prompt or "", translate)

                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures = [
                        executor.submit(_run_transcription, 0, ch1_proc),
                        executor.submit(_run_transcription, 1, ch2_proc),
                    ]
                    results = {}
                    for future in futures:
                        channel_id, segments_out, info_out = future.result()
                        results[channel_id] = (segments_out, info_out)

                ch1_segments, info1 = results[0]
                ch2_segments, info2 = results[1]

                for s in ch1_segments:
                    s["speaker"] = "SPEAKER_00"
                    for w in s["words"]:
                        w["speaker"] = "SPEAKER_00"

                for s in ch2_segments:
                    s["speaker"] = "SPEAKER_01"
                    for w in s["words"]:
                        w["speaker"] = "SPEAKER_01"

                print(f"DEBUG --> Merging segments")
                # all_segments = sorted(ch1_segments + ch2_segments, key=lambda x: x["start"])
                all_segments = self.merge_stereo_words(ch1_segments, ch2_segments)

                detected_language = info1.language or info2.language
                metrics = self._build_metrics(temp_input, all_segments, num_channels=num_channels)
                return Output(all_segments, metrics, language=detected_language, num_speakers=2)

        except Exception as e:
            raise RuntimeError(f"Error running inference: {e}") from e

        finally:
            try:
                cleanup_candidates = {
                    locals().get("temp_input"),
                    locals().get("temp_processed"),
                    locals().get("ch1_path"),
                    locals().get("ch2_path"),
                    locals().get("ch1_proc"),
                    locals().get("ch2_proc"),
                }
                for f in cleanup_candidates:
                    if f and os.path.exists(f):
                        try:
                            os.remove(f)
                        except Exception:
                            pass
            except Exception:
                pass
            try:

                if temp_dir and os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
                # if 'temp_dir' in locals() and temp_dir:
                #     temp_dir.cleanup()
            except Exception:
                pass

    def speech_to_text(
        self,
        audio_file_wav: str,
        num_speakers: Optional[int] = None,
        prompt: str = "",
        language: Optional[str] = None,
        translate: bool = False,
    ) -> Tuple[List[Dict], int, str]:
        segments, transcript_info = self._transcribe_audio(
            audio_file_wav, language, prompt, translate
        )
        print(f"DEBUG --> Finished transcribing, {len(segments)} segments")

        print("DEBUG --> Starting diarization")
        diarization, detected_num_speakers = self._diarize_audio(
            audio_file_wav, num_speakers
        )
        print(f"DEBUG --> Finished diarization, {detected_num_speakers} speakers detected")
        
        print("DEBUG --> Starting merging segments with speaker info")
        final_segments = self._merge_segments_with_diarization(segments, diarization)
        print("DEBUG --> Segments merged and cleaned")

        return final_segments, detected_num_speakers, transcript_info.language

    def _transcribe_audio(self, audio_file_wav, language, prompt, translate):
        options = dict(
            language=language,
            beam_size=5,
            vad_filter=True,
            vad_parameters=VadOptions(
                max_speech_duration_s=self.model.feature_extractor.chunk_length,
                min_speech_duration_ms=50,
                speech_pad_ms=300,
                threshold=0.15,
                neg_threshold=0.05,
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
        return segments, transcript_info
 
    def _diarize_audio(self, audio_file_wav, num_speakers=None):
        waveform, sample_rate = torchaudio.load(audio_file_wav)
        output = self.diarization_model(
            {"waveform": waveform, "sample_rate": sample_rate},
            num_speakers=num_speakers,
        )

        diarize_segments = []
        diarization_list = list(output.speaker_diarization)
        for turn, speaker in diarization_list:
            diarize_segments.append(
                {"start": turn.start, "end": turn.end, "speaker": speaker}
            )

        unique_speakers = {speaker for _, speaker in diarization_list}
        detected_num_speakers = len(unique_speakers)

        return diarize_segments, detected_num_speakers

    def _assign_speaker_to_segment_or_word(self, segment_or_word, diarize_df, fallback_speaker=None):
        """Calculates the intersection of times."""
        diarize_df["intersection"] = np.minimum(
            diarize_df["end"], segment_or_word["end"]
        ) - np.maximum(diarize_df["start"], segment_or_word["start"])
        dia_tmp = diarize_df[diarize_df["intersection"] > 0]

        if len(dia_tmp) > 0:
            speaker = dia_tmp.groupby("speaker")["intersection"].sum().sort_values(ascending=False).index[0]
        else:
            speaker = fallback_speaker or "UNKNOWN"
        return speaker

    def _merge_segments_with_diarization(self, segments, diarize_segments):
        diarize_df = pd.DataFrame(diarize_segments)
        final_segments = []

        for segment in segments:
            speaker = self._assign_speaker_to_segment_or_word(segment, diarize_df)

            words_with_speakers = []
            for word in segment["words"]:
                word_speaker = self._assign_speaker_to_segment_or_word(word, diarize_df, fallback_speaker=speaker)
                word["speaker"] = word_speaker
                words_with_speakers.append(word)

            new_segment = {
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"],
                "speaker": speaker,
                "avg_logprob": segment["avg_logprob"],
                "words": words_with_speakers,
            }
            final_segments.append(new_segment)

        final_segments = self._group_segments(final_segments)
        for segment in final_segments:
            segment["text"] = re.sub(r"\s+", " ", segment["text"]).strip()
            segment["text"] = re.sub(r"\s+([.,!?])", r"\1", segment["text"])
            segment["text"] = self._normalize_currency_in_text(segment["text"])
            segment["duration"] = segment["end"] - segment["start"]

        return final_segments

    def _group_segments(self, segments):
        if not segments:
            return []

        grouped_segments = []
        current_group = segments[0].copy()
        sentence_end_pattern = r"[.!?]+"

        for segment in segments[1:]:
            time_gap = segment["start"] - current_group["end"]
            current_duration = current_group["end"] - current_group["start"]
            can_combine = (
                segment["speaker"] == current_group["speaker"]
                and time_gap <= 1.0
                and current_duration < 30.0
                and not re.search(sentence_end_pattern, current_group["text"][-1:])
            )
            if can_combine:
                current_group["end"] = segment["end"]
                current_group["text"] += " " + segment["text"]
                # current_group["words"] = current_group.get("words", []) + segment.get("words", [])
            else:
                grouped_segments.append(current_group)
                current_group = segment.copy()

        grouped_segments.append(current_group)
        return grouped_segments

    def _get_audio_channels(self, file_path: str) -> int:
        """Identify the number of audio channels using ffprobe."""
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=channels", "-of", "default=noprint_wrappers=1:nokey=1", file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        return int(result.stdout.strip())

    def _split_stereo_channels(self, file_path: str, temp_dir: str) -> Tuple[str, str]:
        """Splits stereo audio into two mono files (left and right channel)."""
        ch1_path = os.path.join(temp_dir, "channel1.wav")
        ch2_path = os.path.join(temp_dir, "channel2.wav")

        subprocess.run([
            "ffmpeg", "-y", "-i", file_path, "-map_channel", "0.0.0",
            "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000", ch1_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        subprocess.run([
            "ffmpeg", "-y", "-i", file_path, "-map_channel", "0.0.1",
            "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000", ch2_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
        return ch1_path, ch2_path
                
    def merge_stereo_words(self, ch1_segments, ch2_segments, overlap_tolerance=0, merge_margin=1):
        words = []
        for seg in ch1_segments + ch2_segments:
            for w in seg["words"]:
                words.append({
                    "start": w["start"],
                    "end": w["end"],
                    "word": w["word"],
                    "speaker": seg["speaker"],
                    "prob": w.get("probability", 1.0),
                })
        words = sorted(words, key=lambda x: x["start"])

        cleaned_words = []
        for i, w in enumerate(words):
            if cleaned_words:
                prev = cleaned_words[-1]
                if w["speaker"] != prev["speaker"] and w["start"] < prev["end"]:
                    overlap = prev["end"] - w["start"]
                    if overlap <= overlap_tolerance:
                        w["start"] = prev["end"] + 0.01
            cleaned_words.append(w)

        merged_segments = []
        current = None
        for w in cleaned_words:
            if not current:
                current = {
                    "start": w["start"],
                    "end": w["end"],
                    "speaker": w["speaker"],
                    "text": w["word"].lstrip(),
                    "words": [w]
                }
                continue

            if (
                w["speaker"] == current["speaker"]
                and w["start"] <= current["end"] + merge_margin
            ):
                current["end"] = w["end"]
                token_text = w["word"]
                if token_text:
                    if token_text[0].isspace():
                        current["text"] += token_text
                    else:
                        current["text"] += " " + token_text
                current["words"].append(w)
            else:
                merged_segments.append(current)
                current = {
                    "start": w["start"],
                    "end": w["end"],
                    "speaker": w["speaker"],
                    "text": w["word"].lstrip(),
                    "words": [w]
                }

        if current:
            merged_segments.append(current)

        for segment in merged_segments:
            text = segment.get("text", "")
            text = re.sub(r"\s+", " ", text).strip()
            text = re.sub(r"\s+([.,!?])", r"\1", text)
            segment["text"] = self._normalize_currency_in_text(text)
            segment["duration"] = segment["end"] - segment["start"]

        return merged_segments

    def _build_metrics(self, audio_path: str, segments: List[Dict], num_channels: Optional[int] = None) -> AudioMetrics:
        audio_duration = SpeechActivityMetricsCalculator.audio_duration_seconds(audio_path)
        calculator = SpeechActivityMetricsCalculator(audio_duration, segments, num_channels=num_channels)
        return calculator.compute()

    def _normalize_currency_in_text(self, text: str) -> str:
        """Normalize currency mentionings to a consistent format.

        Examples normalized:
        - "R $ 346, 90" -> "R$ 346,90"
        - "R$346,90" -> "R$ 346,90" (ensure a space after symbol)

        This is applied only to the final text output (post-processing) and
        does not affect internal diarization or word-level timestamps.
        """
        if not text:
            return text

        # Pattern: R [optional spaces] $ [optional spaces] number (with dots/spaces) decimal-sep fraction
        def repl(m):
            integer = m.group(1) or ""
            frac = m.group(2) or "00"
            # remove spaces inside integer part
            integer_clean = integer.replace(" ", "")
            # keep existing dots (thousand separators) but also remove stray spaces
            integer_clean = integer_clean
            return f"R$ {integer_clean},{frac}"

        # Normalize common variants where decimal uses , or . and there may be spaces
        text = re.sub(r"R\s*\$\s*([0-9\.\s]+)[,\.]\s*([0-9]{2})", repl, text)

        # If pattern like R$346,90 (no space), add a space after R$
        text = re.sub(r"R\$(\d)", r"R$ \1", text)

        return text
