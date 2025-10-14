import os
import re
import argparse
import numpy as np
import librosa
import pandas as pd
from pesq import pesq
from pystoi.stoi import stoi

def mse(x, y):
    return np.mean((x - y) ** 2)

def snr(x, y):
    noise = x - y
    return 10 * np.log10(np.sum(x ** 2) / np.sum(noise ** 2))

def extract_id(filename):
    name, _ = os.path.splitext(filename)
    match = re.search(r"\d+", name)
    return match.group(0) if match else None

def get_audio_files(directory):
    mapping = {}
    for f in os.listdir(directory):
        if f.endswith(".wav"):
            file_id = extract_id(f)
            if file_id:
                mapping[file_id] = os.path.join(directory, f)
    return mapping

def load_audio(path, target_sr=16000):
    audio, sr = librosa.load(path, sr=target_sr, mono=True)
    return audio

def main(original_folder, marked_folder, sample_rate=16000, output_file="results.xlsx"):
    original_files = get_audio_files(original_folder)
    marked_files = get_audio_files(marked_folder)
    common_keys = sorted(set(original_files.keys()) & set(marked_files.keys()))

    print(f"Found {len(common_keys)} pairs to compare.\n") 

    results = []

    for key in common_keys:
        path_orig = original_files[key]
        path_marked = marked_files[key]

        audio_orig = load_audio(path_orig, sample_rate)
        audio_marked = load_audio(path_marked, sample_rate)

        min_len = min(len(audio_orig), len(audio_marked))
        audio_orig = audio_orig[:min_len]
        audio_marked = audio_marked[:min_len]

        val_mse = mse(audio_orig, audio_marked)
        val_snr = snr(audio_orig, audio_marked)
        val_stoi = stoi(audio_orig, audio_marked, sample_rate, extended=False)
        try:
            val_pesq = pesq(sample_rate, audio_orig, audio_marked, 'wb')
        except Exception as e:
            print(f"[!] Error PESQ on {key}: {e}")
            val_pesq = np.nan

        print(f"{key} - MSE: {val_mse:.4f}, SNR: {val_snr:.2f} dB, STOI: {val_stoi:.4f}, PESQ: {val_pesq:.4f}")

        results.append({
            "file": key,
            "mse": val_mse,
            "snr": val_snr,
            "stoi": val_stoi,
            "pesq": val_pesq
        })

    df = pd.DataFrame(results)
    df.to_excel(output_file, index=False)
    print(f"\n✅ Results saved in '{output_file}'")
    print(f"📂 Full path: {os.path.abspath(output_file)}") 

    # Average metrics
    avg_mse = df["mse"].mean()
    avg_snr = df["snr"].mean()
    avg_stoi = df["stoi"].mean()
    avg_pesq = df["pesq"].mean(skipna=True)

    print("\n📊 Average values:")
    print(f"   average MSE  : {avg_mse:.4f}")
    print(f"   average SNR  : {avg_snr:.2f} dB")
    print(f"   average STOI : {avg_stoi:.4f}")
    print(f"   average PESQ : {avg_pesq:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quality metrics evaluation")
    parser.add_argument("--orig", required=True, help="Folder with original audio files")
    parser.add_argument("--marked", required=True, help="Folder with marked (watermarked) audio files")
    parser.add_argument("--sr", type=int, default=16000, help="Target sample rate (default=16000)")
    parser.add_argument("--out", type=str, default="results.xlsx", help="Output Excel filename")
    args = parser.parse_args()

    main(args.orig, args.marked, args.sr, args.out)
