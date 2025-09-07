
"""Cog predictor that wraps the shared pipeline."""
from __future__ import annotations
from typing import Optional
from cog import BasePredictor, BaseModel, Input, Path
from pipeline import WhisperDiarizationPipeline, Output as LocalOutput

class Output(BaseModel):
    segments: list
    language: str = None
    num_speakers: int = None

class Predictor(BasePredictor):
    def setup(self):
        self.pipeline = WhisperDiarizationPipeline(device="cuda", compute_type="float16")
        #self.pipeline = WhisperDiarizationPipeline(device="cpu", compute_type="int8")

    def predict(
        self,
        file_string: Optional[str] = Input(default=None, description="Base64 audio"),
        file_url: Optional[str] = Input(default=None, description="Direct URL"),
        file_path: Path = Input(default=None, description="Audio file"),
        num_speakers: int = Input(default=None, ge=1, le=50, description="Leave empty to autodetect"),
        translate: bool = Input(default=False, description="Translate to English"),
        language: Optional[str] = Input(description="Language code like 'en', 'pt'", default=None),
        prompt: Optional[str] = Input(description="Names/acronyms, separated by punctuation", default=None),
        preprocess: int = Input(default=0, ge=0, le=4, description="0=None, 1=Sanitize, 2=+Filter, 3=+ReduceNoise, 4=+Normalize"),
        highpass_freq: int = Input(default=45),
        lowpass_freq: int = Input(default=8000),
        prop_decrease: float = Input(default=0.3),
        stationary: bool = Input(default=True),
        target_dBFS: float = Input(default=-18.0),
    ) -> Output:
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
        )