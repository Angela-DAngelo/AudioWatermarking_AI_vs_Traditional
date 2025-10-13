#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Unified Audio Perturbation Script
---------------------------------
This script merges two pipelines:
  1) Classic DSP perturbations (noise, filters, echo, quantization, smoothing, time-stretch, mp3)
  2) Codec-based perturbations (OPUS via ffmpeg/pydub, Encodec via HuggingFace)

Usage examples:
  python apply_all_audio_perturbations.py --input input_audio --output output_audio --which all
  python apply_all_audio_perturbations.py --which basic
  python apply_all_audio_perturbations.py --which codec --opus-bitrates 16 32 64 --encodec-bw 1.5 3 6

Requirements:
  - torch, torchaudio, numpy, librosa, julius, pydub, soundfile, tqdm
  - ffmpeg installed on your system (for OPUS/MP3 encoding through pydub)
  - transformers (for Encodec): facebook/encodec_24khz model will be downloaded on first use

Outputs are saved as WAV at the target sample rate (default 44100 Hz).
"""
import os
import argparse
import warnings
import uuid
import tempfile

import torch
import torchaudio
from tqdm import tqdm
import numpy as np
import julius

from pydub import AudioSegment
import soundfile as sf
import librosa

# Optional Encodec imports (loaded only if needed)
try:
    from transformers import EncodecModel, AutoProcessor
    _TRANSFORMERS_AVAILABLE = True
except Exception:
    _TRANSFORMERS_AVAILABLE = False


# ============================
# Helpers: I/O & Resampling
# ============================
def _to_mono_and_resample(waveform: torch.Tensor, sr: int, target_sr: int = 44100):
    # to mono
    if waveform.ndim == 2 and waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    elif waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    # resample
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(waveform)
        sr = target_sr
    return waveform, sr


# ============================
# DSP Perturbations
# ============================
def pert_time_stretch(waveform, rate):
    waveform_np = waveform.numpy().squeeze()
    stretched = librosa.effects.time_stretch(waveform_np, rate=rate)
    return torch.from_numpy(stretched).unsqueeze(0).float()

def pert_gaussian_noise(waveform, snr_db):
    signal_power = torch.mean(waveform**2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
    return waveform + noise

def pert_background_noise(waveform, snr_db):
    # Uses a VOiCES sample shipped with torchaudio tutorial assets
    noise, _ = torchaudio.load(torchaudio.utils.download_asset(
        "tutorial-assets/Lab41-SRI-VOiCES-rm1-babb-mc01-stu-clo-8000hz.wav"))
    if noise.size(1) < waveform.size(1):
        reps = (waveform.size(1) // noise.size(1)) + 1
        noise = noise.repeat(1, reps)[:, :waveform.size(1)]
    signal_power = torch.mean(waveform**2)
    noise_power = torch.mean(noise**2)
    scaling = torch.sqrt(signal_power / (10 ** (snr_db / 10) * noise_power))
    return waveform + noise * scaling

def pert_lowpass(waveform, cutoff):
    return julius.lowpass_filter(waveform, cutoff=cutoff)

def pert_highpass(waveform, cutoff):
    return julius.highpass_filter(waveform, cutoff=cutoff)

def pert_echo(waveform, delay, volume=0.4, sample_rate=44100):
    # waveform: (1, T)
    x = waveform.unsqueeze(0)  # (B=1, C=1, T)
    n_samples = int(sample_rate * delay)
    impulse = torch.zeros(n_samples, dtype=x.dtype, device=x.device)
    impulse[0] = 1.0
    impulse[-1] = volume
    impulse = impulse.unsqueeze(0).unsqueeze(0)  # (B=1, C=1, K)
    result = julius.fft_conv1d(x, impulse)
    # normalize to original peak
    max_x = torch.max(torch.abs(x))
    max_r = torch.max(torch.abs(result))
    if max_r > 0:
        result = result / max_r * max_x
    return result.squeeze(0)

def pert_quantization(waveform, levels):
    min_val, max_val = waveform.min(), waveform.max()
    if max_val - min_val <= 1e-12:
        return waveform.clone()
    norm = (waveform - min_val) / (max_val - min_val)
    quantized = torch.round(norm * (levels - 1))
    return (quantized / (levels - 1)) * (max_val - min_val) + min_val

def pert_smooth(waveform, window_size):
    x = waveform.unsqueeze(0)  # (B=1, C=1, T)
    kernel = torch.ones(1, 1, window_size, dtype=x.dtype, device=x.device) / window_size
    smoothed = julius.fft_conv1d(x, kernel)
    return smoothed.squeeze(0)

def pert_mp3(waveform, sample_rate, bitrate):
    tmp_dir = tempfile.gettempdir()
    tmp_wav = os.path.join(tmp_dir, f"tmp_{uuid.uuid4().hex}.wav")
    tmp_mp3 = os.path.join(tmp_dir, f"tmp_{uuid.uuid4().hex}.mp3")
    try:
        sf.write(tmp_wav, waveform.squeeze().cpu().numpy(), samplerate=sample_rate)
        audio = AudioSegment.from_wav(tmp_wav)
        audio.export(tmp_mp3, format="mp3", bitrate=f"{bitrate}k")
        pert, _ = librosa.load(tmp_mp3, sr=sample_rate)
        return torch.tensor(pert).unsqueeze(0)
    finally:
        for p in (tmp_wav, tmp_mp3):
            if os.path.exists(p):
                try: os.remove(p)
                except Exception: pass


# ============================
# Codec Perturbations
# ============================
def pert_opus_ffmpeg(waveform, sample_rate=44100, bitrate_kbps=24):
    tmp_dir = tempfile.gettempdir()
    tmp_wav = os.path.join(tmp_dir, f"temp_{uuid.uuid4().hex}.wav")
    tmp_opus = os.path.join(tmp_dir, f"temp_{uuid.uuid4().hex}.opus")
    try:
        sf.write(tmp_wav, waveform.squeeze().cpu().numpy(), samplerate=sample_rate)
        audio = AudioSegment.from_wav(tmp_wav)
        audio.export(tmp_opus, format="opus", bitrate=f"{bitrate_kbps}k")
        pert, _ = librosa.load(tmp_opus, sr=sample_rate)
        return torch.tensor(pert).unsqueeze(0)
    finally:
        for p in (tmp_wav, tmp_opus):
            if os.path.exists(p):
                try: os.remove(p)
                except Exception: pass

def _load_encodec():
    if not _TRANSFORMERS_AVAILABLE:
        raise RuntimeError("transformers not installed; cannot use Encodec. `pip install transformers`")
    model = EncodecModel.from_pretrained("facebook/encodec_24khz")
    processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    return model, processor, device

def pert_encodec(waveform, sample_rate, bandwidth, model_encodec, processor, device):
    # Encodec expects 24kHz mono
    if sample_rate != 24000:
        waveform_24 = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=24000)(waveform)
        sr_in = 24000
    else:
        waveform_24 = waveform
        sr_in = sample_rate

    # HF processor expects numpy
    w_np = waveform_24.squeeze().cpu().numpy()
    inputs = processor(raw_audio=w_np, sampling_rate=sr_in, return_tensors="pt").to(device)

    with torch.no_grad():
        enc_out = model_encodec.encode(inputs["input_values"], inputs["padding_mask"], bandwidth)
        decoded = model_encodec.decode(enc_out.audio_codes, enc_out.audio_scales, inputs["padding_mask"])[0]

    decoded_tensor = torch.tensor(decoded.cpu()).squeeze().unsqueeze(0)
    # back to 44.1k if needed
    return torchaudio.transforms.Resample(orig_freq=24000, new_freq=44100)(decoded_tensor)


# ============================
# Pipelines
# ============================
def apply_basic_perturbations(input_dir, output_dir, sample_rate=44100):
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.wav', '.mp3'))]

    configs = {
        "time_stretch": [0.7, 0.9, 1.1, 1.3, 1.5],
        "gaussian_noise": [5, 10, 20, 30, 40],
        "background_noise": [5, 10, 20, 30, 40],
        "lowpass": [0.1, 0.2, 0.3, 0.4, 0.5],
        "highpass": [0.06, 0.07, 0.08, 0.09, 0.1],
        "echo": [0.1, 0.3, 0.5, 0.7, 0.9],
        "quantization": [4, 6, 8, 12, 16],
        "smooth": [6, 10, 14, 18, 22],
        "mp3": [8, 16, 24, 32, 40]
    }

    for filename in tqdm(files, desc="Applying basic DSP perturbations"):
        path = os.path.join(input_dir, filename)
        waveform, sr = torchaudio.load(path)
        waveform, sr = _to_mono_and_resample(waveform, sr, sample_rate)
        basename = os.path.splitext(filename)[0]

        for name, params in configs.items():
            for param in params:
                try:
                    if name == "time_stretch":
                        pert = pert_time_stretch(waveform, param)
                    elif name == "gaussian_noise":
                        pert = pert_gaussian_noise(waveform, param)
                    elif name == "background_noise":
                        pert = pert_background_noise(waveform, param)
                    elif name == "lowpass":
                        pert = pert_lowpass(waveform, param)
                    elif name == "highpass":
                        pert = pert_highpass(waveform, param)
                    elif name == "echo":
                        pert = pert_echo(waveform, delay=param, sample_rate=sr)
                    elif name == "quantization":
                        pert = pert_quantization(waveform, levels=param)
                    elif name == "smooth":
                        pert = pert_smooth(waveform, window_size=param)
                    elif name == "mp3":
                        pert = pert_mp3(waveform, sample_rate=sr, bitrate=param)
                    else:
                        continue

                    save_path = os.path.join(output_dir, f"{basename}_{name}_{param}.wav")
                    torchaudio.save(save_path, pert, sr)
                except Exception as e:
                    print(f"[WARN] DSP {name} ({param}) failed on {filename}: {e}")

def apply_codec_perturbations(input_dir, output_dir, sample_rate=44100,
                              opus_bitrates=(16, 32, 64, 128, 256),
                              encodec_bandwidths=(1.5, 3, 6, 12, 24),
                              use_encodec=True, use_opus=True):
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.wav', '.mp3'))]

    # Lazy-load Encodec if requested
    model_encodec = processor = device = None
    if use_encodec:
        try:
            model_encodec, processor, device = _load_encodec()
        except Exception as e:
            warnings.warn(f"Encodec unavailable: {e}. Skipping Encodec perturbations.")
            use_encodec = False

    for filename in tqdm(files, desc="Applying codec perturbations"):
        path = os.path.join(input_dir, filename)
        waveform, sr = torchaudio.load(path)
        waveform, sr = _to_mono_and_resample(waveform, sr, sample_rate)
        basename = os.path.splitext(filename)[0]

        if use_opus:
            for br in opus_bitrates:
                try:
                    pert_op = pert_opus_ffmpeg(waveform, sample_rate=sr, bitrate_kbps=br)
                    out_path = os.path.join(output_dir, f"{basename}_opus_{br}kbps.wav")
                    torchaudio.save(out_path, pert_op, sr)
                except Exception as e:
                    print(f"[WARN] OPUS {br}kbps failed on {filename}: {e}")

        if use_encodec and model_encodec is not None:
            for bw in encodec_bandwidths:
                try:
                    pert_enc = pert_encodec(waveform, sr, bandwidth=bw,
                                            model_encodec=model_encodec, processor=processor, device=device)
                    out_path = os.path.join(output_dir, f"{basename}_encodec_{bw}bw.wav")
                    torchaudio.save(out_path, pert_enc, 44100)
                except Exception as e:
                    print(f"[WARN] Encodec {bw}bw failed on {filename}: {e}")


# ============================
# Entry point
# ============================
def main():
    parser = argparse.ArgumentParser(description="Apply DSP and codec audio perturbations.")
    parser.add_argument("--input", "-i", default="input_audio", help="Input directory with audio files (.wav/.mp3)")
    parser.add_argument("--output", "-o", default="output_audio", help="Output directory for perturbed WAV files")
    parser.add_argument("--which", choices=["basic", "codec", "all"], default="all",
                        help="Which perturbations to run")
    parser.add_argument("--sr", type=int, default=44100, help="Target sample rate for processing/saving")
    parser.add_argument("--opus-bitrates", type=int, nargs="*", default=[16, 32, 64, 128, 256],
                        help="OPUS bitrates (kbps)")
    parser.add_argument("--encodec-bw", type=float, nargs="*", default=[1.5, 3, 6, 12, 24],
                        help="Encodec bandwidths")
    parser.add_argument("--no-opus", action="store_true", help="Disable OPUS perturbations")
    parser.add_argument("--no-encodec", action="store_true", help="Disable Encodec perturbations")

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.which in ("basic", "all"):
        apply_basic_perturbations(args.input, args.output, sample_rate=args.sr)

    if args.which in ("codec", "all"):
        apply_codec_perturbations(
            args.input,
            args.output,
            sample_rate=args.sr,
            opus_bitrates=tuple(args.opus_bitrates),
            encodec_bandwidths=tuple(args.encodec_bw),
            use_encodec=not args.no_encodec,
            use_opus=not args.no_opus,
        )

if __name__ == "__main__":
    main()
