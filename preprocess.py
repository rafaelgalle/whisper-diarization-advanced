"""Audio preprocessing utilities."""
import os
import subprocess
import numpy as np
from scipy.io import wavfile
from scipy.io.wavfile import write
import noisereduce as nr
import tempfile
import shutil

def sanitize_audio(input_path: str, sanitized_path: str):
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
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"DEBUG --> Sanitization completed: {sanitized_path}")


def apply_filters(input_path: str, filtered_path: str, highpass_freq: int, lowpass_freq: int):
    """Applies highpass and lowpass filters via ffmpeg."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-af", f"highpass=f={highpass_freq},lowpass=f={lowpass_freq}",
        filtered_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"DEBUG --> Applied filters: {filtered_path}")


def reduce_noise(input_path: str, prop_decrease: float, stationary: bool):
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
    print("DEBUG --> Noise removed")
    return rate, reduced


def normalize_audio(audio_data: np.ndarray, target_dBFS: float):
    """Normalizes audio RMS to target dBFS."""
    rms = np.sqrt(np.mean(audio_data**2))
    target_rms = 10 ** (target_dBFS / 20)
    gain = target_rms / (rms + 1e-9)
    print("DEBUG --> Audio normalized")
    return audio_data * gain


def preprocess_audio(
    input_path: str,
    output_path: str,
    preprocess_level: int = 4,
    highpass_freq: int = 45,
    lowpass_freq: int = 8000,
    prop_decrease: float = 1.0,
    stationary: bool = False,
    target_dBFS: float = -20.0,
):
    """
    Performs audio cleanup to the specified level.

    Levels:
    0 - Does nothing (copies the original file)
    1 - Sanitization
    2 - Sanitization + Filter
    3 - Sanitization + Filter + ReduceNoise
    4 - Sanitization + Filter + ReduceNoise + Normalization
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        temp1 = f"{tmpdir}/stage1.wav"
        temp2 = f"{tmpdir}/stage2.wav"
        temp3 = f"{tmpdir}/stage3.wav"

        if preprocess_level == 0:
            shutil.copy(input_path, output_path)
            return

        sanitize_audio(input_path, temp1)
        if preprocess_level == 1:
            shutil.copy(temp1, output_path)
            return

        apply_filters(temp1, temp2, highpass_freq, lowpass_freq)
        if preprocess_level == 2:
            shutil.copy(temp2, output_path)
            return

        rate, reduced = reduce_noise(temp2, prop_decrease, stationary)
        out_int16 = np.clip(reduced * 32768, -32768, 32767).astype(np.int16)
        write(temp3, rate, out_int16)
        if preprocess_level == 3:
            shutil.copy(temp3, output_path)
            return

        normalized = normalize_audio(reduced, target_dBFS)
        out_int16 = np.clip(normalized * 32768, -32768, 32767).astype(np.int16)
        write(output_path, rate, out_int16)

        # debug_folder = "/home/rafaelgalle/Downloads/audio_debug"
        # os.makedirs(debug_folder, exist_ok=True)
        # # shutil.copy(temp1, f"{debug_folder}/stage1.wav")
        # # shutil.copy(temp2, f"{debug_folder}/stage2.wav")
        # # shutil.copy(temp3, f"{debug_folder}/stage3.wav")
        # shutil.copy(output_path, f"{debug_folder}/final_output_{output_path.split('/')[-1].replace('.wav', '')}.wav")
        # print(f"DEBUG --> Copy files to: {debug_folder}")

        print(f"DEBUG --> Preprocessing complete. File saved in: {output_path}")
