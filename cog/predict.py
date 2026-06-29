"""Cog predictor for Whisper + Diarization pipeline."""
from __future__ import annotations
from typing import Optional
from cog import BasePredictor, BaseModel, Input, Path
from core.pipeline import WhisperDiarizationPipeline, Output as LocalOutput


class Output(BaseModel):
    """Output format for Cog."""
    segments: list
    language: str = None
    num_speakers: int = None
    metrics: dict = None


class Predictor(BasePredictor):
    def setup(self):
        """Load models into memory once at startup."""
        self.pipeline = WhisperDiarizationPipeline(
            device="cuda",
            compute_type="float16"
            # device="cpu",
            # compute_type="int8"
        )

    def predict(
        self,
        file_path: Optional[Path] = Input(default=None, description="Audio file path"),
        file_url: Optional[str] = Input(default=None, description="Direct URL to audio"),
        file_string: Optional[str] = Input(default=None, description="Base64 encoded audio"),
        num_speakers: Optional[int] = Input(default=None, ge=1, le=50, description="Number of speakers (leave empty to auto-detect)"),
        translate: bool = Input(default=False, description="Translate to English"),
        language: Optional[str] = Input(description="Language code (e.g., 'en', 'pt')", default=None),
        prompt: Optional[str] = Input(description="Custom prompt with names/acronyms separated by punctuation", default=None),
        preprocess: int = Input(default=0, ge=0, le=4, description="Preprocessing level: 0=None, 1=Sanitize, 2=+Filter, 3=+ReduceNoise, 4=+Normalize"),
        highpass_freq: int = Input(default=45, description="Highpass filter frequency (Hz)"),
        lowpass_freq: int = Input(default=8000, description="Lowpass filter frequency (Hz)"),
        prop_decrease: float = Input(default=0.3, ge=0.0, le=1.0, description="Noise reduction proportion"),
        stationary: bool = Input(default=True, description="Use stationary noise reduction"),
        target_dBFS: float = Input(default=-18.0, description="Target loudness in dBFS"),
    ) -> Output:
        """Run transcription and diarization on audio."""
        result: LocalOutput = self.pipeline.predict(
            file_string=file_string,
            file_url=file_url,
            file_path=str(file_path) if file_path else None,
            num_speakers=num_speakers,
            translate=translate,
            language=language,
            prompt=prompt,
            preprocess=preprocess,
            highpass_freq=highpass_freq,
            lowpass_freq=lowpass_freq,
            prop_decrease=prop_decrease,
            stationary=stationary,
            target_dBFS=target_dBFS,
        )
        
        return Output(
            segments=result.segments,
            language=result.language,
            num_speakers=result.num_speakers,
            metrics=result.metrics.to_dict() if result.metrics else {},
        )
