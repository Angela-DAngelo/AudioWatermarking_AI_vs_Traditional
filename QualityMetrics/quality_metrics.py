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

def main(cartella_originali, cartella_marchiati, sample_rate=16000, output_file="risultati_metriche.xlsx"):
    original_files = get_audio_files(cartella_originali)
    marchiati_files = get_audio_files(cartella_marchiati)
    chiavi_comuni = sorted(set(original_files.keys()) & set(marchiati_files.keys()))

    print(f"Trovate {len(chiavi_comuni)} coppie di file audio da confrontare.\n") 

    risultati = []

    for chiave in chiavi_comuni:
        path_orig = original_files[chiave]
        path_marc = marchiati_files[chiave]

        audio_orig = load_audio(path_orig, sample_rate)
        audio_marc = load_audio(path_marc, sample_rate)

        min_len = min(len(audio_orig), len(audio_marc))
        audio_orig = audio_orig[:min_len]
        audio_marc = audio_marc[:min_len]

        val_mse = mse(audio_orig, audio_marc)
        val_snr = snr(audio_orig, audio_marc)
        val_stoi = stoi(audio_orig, audio_marc, sample_rate, extended=False)
        try:
            val_pesq = pesq(sample_rate, audio_orig, audio_marc, 'wb')
        except Exception as e:
            print(f"[!] Errore PESQ su {chiave}: {e}")
            val_pesq = np.nan

        print(f"{chiave} - MSE: {val_mse:.4f}, SNR: {val_snr:.2f} dB, STOI: {val_stoi:.4f}, PESQ: {val_pesq:.4f}")

        risultati.append({
            "file": chiave,
            "mse": val_mse,
            "snr": val_snr,
            "stoi": val_stoi,
            "pesq": val_pesq
        })

    df = pd.DataFrame(risultati)
    df.to_excel(output_file, index=False)
    print(f"\n✅ Risultati salvati in '{output_file}'")
    print(f"📂 Percorso completo: {os.path.abspath(output_file)}") 

    # Calcolo e stampa medie complessive
    media_mse = df["mse"].mean()
    media_snr = df["snr"].mean()
    media_stoi = df["stoi"].mean()
    media_pesq = df["pesq"].mean(skipna=True)

    print("\n📊 Medie complessive:")
    print(f"   MSE medio : {media_mse:.4f}")
    print(f"   SNR medio : {media_snr:.2f} dB")
    print(f"   STOI medio: {media_stoi:.4f}")
    print(f"   PESQ medio: {media_pesq:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calcolo metriche di qualità tra audio originali e marchiati")
    parser.add_argument("--orig", required=True, help="Cartella contenente i file audio originali")
    parser.add_argument("--marchiati", required=True, help="Cartella contenente i file audio marchiati")
    parser.add_argument("--sr", type=int, default=16000, help="Sample rate target (default=16000)")
    parser.add_argument("--out", type=str, default="risultati_metriche.xlsx", help="Nome file Excel di output")
    args = parser.parse_args()

    main(args.orig, args.marchiati, args.sr, args.out)
