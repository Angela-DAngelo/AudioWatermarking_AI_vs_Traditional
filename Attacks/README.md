## ⚙️ apply_audio_perturbations.py

### Description
`apply_audio_perturbations.py` is a unified script for applying a wide range of **audio perturbations** to evaluate the robustness of watermarking systems or other audio processing pipelines.  
It combines two categories of transformations:

1. **Classic DSP perturbations** — such as additive noise, filtering, echo, quantization, smoothing, and time-stretching.  
2. **Codec-based perturbations** — including **MP3**, **OPUS**, and **Encodec** neural codec compression.

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

Apply all perturbations:
```
python apply_audio_perturbations.py --input input_audio --output output_audio --which all
```
Apply only classic DSP perturbations:
```
python apply_audio_perturbations.py --which basic
```
Apply only codec perturbations (OPUS + Encodec):
```
python apply_audio_perturbations.py --which codec --opus-bitrates 16 32 64 --encodec-bw 1.5 3 6
```

### Perturbation Types

**DSP-based**:  
Time-stretching  
Gaussian noise  
Background noise  
Low-pass and high-pass filtering  
Echo addition  
Quantization  
Smoothing  
MP3 compression  

**Codec-based**:  
OPUS (bitrate variants: 16–256 kbps)  
Encodec (bandwidth variants: 1.5–24 kHz, using facebook/encodec_24khz)  

### Output

Perturbed audio files are saved in the specified output directory with filenames of the form:
```
<basename>_<perturbation>_<parameter>.wav
```

Example:
```
speech_sample_gaussian_noise_20.wav
speech_sample_opus_64kbps.wav
speech_sample_encodec_6bw.wav
```
