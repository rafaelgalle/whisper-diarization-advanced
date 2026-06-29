from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import torchaudio


@dataclass
class AudioMetrics:
    total_audio_duration: float
    speaker0_talk_duration: float
    speaker1_talk_duration: float
    speaker0_silence_duration: float
    speaker1_silence_duration: float
    any_speaker_talk_duration: float
    no_speaker_silence_duration: float
    no_speaker_apparent_silence_duration: float
    num_channels: int = 0
    channel_type: str = "unknown"

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


class SpeechActivityMetricsCalculator:
    """Utility class to compute speech activity metrics for diarized audio."""

    def __init__(
        self,
        total_duration: float,
        segments: Optional[List[Dict]] = None,
        min_silence_duration: float = 4.0,
        num_channels: Optional[int] = None,
    ):
        self.total_duration = max(float(total_duration), 0.0)
        self.segments = segments or []
        self._speaker_assignments: Dict[str, int] = {}
        self.min_silence_duration = max(float(min_silence_duration), 0.0)
        self.num_channels = int(num_channels) if num_channels is not None else 0

    @staticmethod
    def audio_duration_seconds(audio_path: str) -> float:
        """Return audio duration in seconds for the given PCM WAV file."""
        if not audio_path:
            return 0.0
        try:
            info = torchaudio.info(audio_path)
            if info.sample_rate and info.sample_rate > 0:
                return info.num_frames / float(info.sample_rate)
        except Exception:
            try:
                waveform, sample_rate = torchaudio.load(audio_path)
                if sample_rate > 0:
                    return waveform.size(-1) / float(sample_rate)
            except Exception:
                return 0.0
        return 0.0

    def compute(self) -> AudioMetrics:
        speaker_intervals: Dict[int, List[Tuple[float, float]]] = {0: [], 1: []}
        speech_intervals: List[Tuple[float, float]] = []

        for segment in self.segments:
            start = float(segment.get("start", 0.0))
            end = float(segment.get("end", start))
            if math.isinf(start) or math.isnan(start) or math.isnan(end):
                continue
            start = max(0.0, start)
            end = max(start, end)
            if self.total_duration > 0:
                end = min(end, self.total_duration)
            if end <= start:
                continue

            speech_intervals.append((start, end))

            speaker_label = segment.get("speaker")
            speaker_idx = self._resolve_speaker_index(speaker_label)
            if speaker_idx in (0, 1):
                speaker_intervals[speaker_idx].append((start, end))

        speaker_talk = {
            idx: self._sum_intervals(intervals)
            for idx, intervals in speaker_intervals.items()
        }
        any_speaker_talk = self._sum_intervals(speech_intervals)

        speaker_silence = {
            idx: max(self.total_duration - duration, 0.0)
            for idx, duration in speaker_talk.items()
        }
        real_silence = max(self.total_duration - any_speaker_talk, 0.0)
        apparent_silence = self._silence_duration(speech_intervals)

        # Channel info
        num_ch = self.num_channels
        if num_ch <= 0:
            channel_type = "unknown"
        elif num_ch == 1:
            channel_type = "mono"
        else:
            channel_type = "stereo"

        return AudioMetrics(
            total_audio_duration=self.total_duration,
            speaker0_talk_duration=speaker_talk.get(0, 0.0),
            speaker1_talk_duration=speaker_talk.get(1, 0.0),
            speaker0_silence_duration=speaker_silence.get(0, self.total_duration),
            speaker1_silence_duration=speaker_silence.get(1, self.total_duration),
            any_speaker_talk_duration=any_speaker_talk,
            no_speaker_silence_duration=real_silence,
            no_speaker_apparent_silence_duration=apparent_silence,
            num_channels=num_ch,
            channel_type=channel_type,
        )

    def _resolve_speaker_index(self, label: Optional[str]) -> Optional[int]:
        if label is None:
            return None
        normalized = str(label).strip()
        if not normalized:
            return None

        match = re.search(r"(\d+)", normalized)
        if match:
            idx = int(match.group(1))
            if idx in (0, 1):
                return idx

        key = normalized.lower()
        if key not in self._speaker_assignments and len(self._speaker_assignments) < 2:
            self._speaker_assignments[key] = len(self._speaker_assignments)

        idx = self._speaker_assignments.get(key)
        if idx in (0, 1):
            return idx
        return None

    def _sum_intervals(self, intervals: List[Tuple[float, float]]) -> float:
        if not intervals:
            return 0.0

        merged = self._merge_intervals(intervals)
        total = sum(max(0.0, end - start) for start, end in merged)

        if self.total_duration > 0:
            return min(total, self.total_duration)
        return total

    def _merge_intervals(self, intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if not intervals:
            return []

        sorted_intervals = sorted(intervals, key=lambda item: item[0])
        merged: List[Tuple[float, float]] = []
        current_start, current_end = sorted_intervals[0]

        for start, end in sorted_intervals[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
            else:
                merged.append((current_start, current_end))
                current_start, current_end = start, end

        merged.append((current_start, current_end))

        if self.total_duration > 0:
            bounded = []
            for start, end in merged:
                bounded_start = max(0.0, min(start, self.total_duration))
                bounded_end = max(bounded_start, min(end, self.total_duration))
                if bounded_end > bounded_start:
                    bounded.append((bounded_start, bounded_end))
            return bounded
        return merged

    def _silence_duration(self, speech_intervals: List[Tuple[float, float]]) -> float:
        if self.total_duration <= 0:
            return 0.0

        merged_speech = self._merge_intervals(speech_intervals)
        silences: List[Tuple[float, float]] = []
        cursor = 0.0

        for start, end in merged_speech:
            if start > cursor:
                silences.append((cursor, start))
            cursor = max(cursor, end)

        if cursor < self.total_duration:
            silences.append((cursor, self.total_duration))

        filtered = [
            (start, end)
            for start, end in silences
            if (end - start) >= self.min_silence_duration
        ]

        return sum(end - start for start, end in filtered)
