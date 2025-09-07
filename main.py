"""Local entry point: python -m app.main --file_path input.wav ..."""
from __future__ import annotations
import argparse
from pipeline import WhisperDiarizationPipeline

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Local runner for Whisper + Diarization")
    p.add_argument("--file_string", type=str, default=None)
    p.add_argument("--file_url", type=str, default=None)
    p.add_argument("--file_path", type=str, default=None)
    p.add_argument("--num_speakers", type=int, default=None)
    p.add_argument("--translate", type=bool, default=False)
    p.add_argument("--language", type=str, default=None)
    p.add_argument("--prompt", type=str, default=None)
    p.add_argument("--preprocess", type=int, default=0, choices=[0,1,2,3,4])
    p.add_argument("--highpass_freq", type=int, default=45)
    p.add_argument("--lowpass_freq", type=int, default=8000)
    p.add_argument("--prop_decrease", type=float, default=0.3)
    p.add_argument("--stationary", type=bool, default=True)
    p.add_argument("--target_dBFS", type=float, default=-18.0)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu","cuda"])
    p.add_argument("--compute_type", type=str, default="int8")
    p.add_argument("--model_name", type=str, default="large-v3-turbo")
    return p

def main():
    args = build_parser().parse_args()
    print(f"DEBUG --> Args: {args}")
    pipeline = WhisperDiarizationPipeline(
        device=args.device,
        compute_type=args.compute_type,
        model_name=args.model_name,
    )

    result = pipeline.predict(
        file_string=args.file_string,
        file_url=args.file_url,
        file_path=args.file_path,
        num_speakers=args.num_speakers,
        translate=args.translate,
        language=args.language,
        prompt=args.prompt,
        preprocess=args.preprocess,
        highpass_freq=args.highpass_freq,
        lowpass_freq=args.lowpass_freq,
        prop_decrease=args.prop_decrease,
        stationary=args.stationary,
        target_dBFS=args.target_dBFS,
    )

    print(result.to_dict())

if __name__ == "__main__":
    main()