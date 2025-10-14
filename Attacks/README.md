## ⚙️ apply_audio_perturbations.py

### Description
`apply_audio_perturbations.py` is a script for applying a wide range of **audio perturbations** to evaluate the robustness of watermarking systems or other audio processing pipelines.  
It combines two categories of transformations (different parameter values are used to generate multiple variants of each attack):

1. **Classic DSP perturbations**, such as additive noise, filtering, echo, quantization, smoothing, and time-stretching.  
2. **Codec-based perturbations**, including **MP3**, **OPUS**, and **Encodec** neural codec compression.

All processed files are saved as `.wav` at the target sample rate (default **44.1 kHz**).

---

### Requirements
Install dependencies:
```bash
pip install torch torchaudio numpy librosa julius pydub soundfile tqdm transformers
```

Additionally, ensure ffmpeg is installed and available in your system path (for MP3/OPUS encoding).

---

### Usage Examples

Run the script from the command line:

Apply all perturbations to an input folder and save the attacked audio in an output folder:
```
python apply_audio_perturbations.py --input input_audio --output output_audio --which all
```
Apply only classic DSP perturbations:
```
python apply_audio_perturbations.py --input input_audio --output output_audio --which basic
```
Apply only codec perturbations (OPUS + Encodec):
```
python apply_audio_perturbations.py --input input_audio --output output_audio --which codec --opus-bitrates 16 32 64 --encodec-bw 1.5 3 6
```

### Details of Implemented Attacks

| **Perturbation**   | **Key Parameter (K)** | **Values of K** |
|--------------------|-----------------------|-----------------|
| Time Stretch       | Speed Factor          | [0.7, 0.8, 1.1, 1.3, 1.5] |
| Gaussian Noise     | SNR (dB)              | [5, 10, 20, 30, 40] |
| Background Noise   | SNR (dB)              | [5, 10, 20, 30, 40] |
| Opus               | Bitrate (kbps)        | [16, 32, 64, 128, 256] |
| EnCodec            | Bandwidth (kHz)       | [1.5, 3, 6, 12, 24.0] |
| Quantization       | Bit Levels            | [4, 6, 8, 12, 16] |
| High-pass Filter   | Cutoff Ratio          | [0.1, 0.2, 0.3, 0.4, 0.5] |
| Low-pass Filter    | Cutoff Ratio          | [0.1, 0.2, 0.3, 0.4, 0.5] |
| Smoothing          | Window Size           | [6, 10, 14, 18, 22] |
| Echo               | Delay (sec)           | [0.1, 0.3, 0.5, 0.7, 0.9] |
| MP3 Compression    | Bitrate (kbps)        | [8, 16, 24, 32, 40] |


### Output

Perturbed audio files are saved in the specified output directory with filenames of the form:
```
<basename>_<perturbation>_<parameter>.wav
```

Example:
```
wm_0000_gaussian_noise_20.wav
wm_0000_opus_64kbps.wav
wm_0000_encodec_6bw.wav
```
