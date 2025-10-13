# 🎵 AudioWatermarking_AI_vs_Traditional

### Comparative Study of AI-based and Traditional Audio Watermarking Methods

This repository accompanies the research work *"Assessing Progress over a Decade of Digital Audio Watermarking Research"* (IEEE Access, 2024).

It provides a comparative framework between classical signal-processing watermarking (STAMP) and modern AI-based approaches such as AudioSeal, WavMark, and SilentCipher.

---

## 💡 Overview

Digital audio watermarking is a core technology for copyright protection, authenticity verification, and synthetic media detection.  
While recent AI-based models have been proposed for watermark embedding and detection, their actual robustness compared to traditional designs remains uncertain.

This project evaluates both paradigms under identical conditions, assessing:
- **Imperceptibility**
- **Robustness**
- **Bit recovery accuracy**

---

## 🤖 Audio Watermarking Systems

- **STAMP**: **S**pectral **T**ransform-domain **A**udio **M**arking with **P**erceptual model (proposed classical system)
- **AudioSeal**: https://github.com/facebookresearch/audioseal (state-of-the-art AI-based system)
- **WavMark**: https://github.com/wavmark/wavmark (state-of-the-art AI-based system)
- **SilentCipher**: https://github.com/sony/silentcipher (state-of-the-art AI-based system)

---

## ⚙️ Evaluation Framework

- **Datasets:** AudioMarkBench, LibriSpeech, FMA  
- **Attacks:** Time Stretch, Gaussian Noise, Background Noise, Opus, EnCodec, Quantization, Highpass filter, Lowpass filter, Smooth, Echo, Mp3 compression 
- **Metrics:**  
  - Bit Recovery Accuracy (ACC)  
  - False Positive / Negative Rates (FPR, FNR)  
  - Audio Quality: SNR, PESQ, STOI, ViSQOL  

---

## 🗂️ Repository Structure

The repository is organized into four main directories:

- **Dataset/** – audio datasets used for benchmarking and evaluation (speech and music).  
- **Systems/** – implementations of both classical and AI-based watermarking systems.  
- **Attacks/** – signal perturbation scripts for robustness testing.  
- **QualityMetrics/** – evaluation tools and scripts for measuring audio quality and imperceptibility (e.g., SNR, PESQ, STOI, ViSQOL).

```text
AudioWatermarking_AI_vs_Traditional/
│
├── Dataset/
│   ├── AudioMarkBench/          # Multilingual speech benchmark
│   ├── LibriSpeech/             # English speech dataset
│   └── FMA/                     # Free Music Archive dataset
│
├── Systems/
│   ├── STAMP/                   # Classical signal-processing system (proposed)
│   ├── AudioSeal/               # AI-based sequence-to-sequence watermarking
│   ├── WavMark/                 # Spectrogram-based neural watermarking
│   └── SilentCipher/            # Deep spectrogram watermarking with psychoacoustic model
│
├── Attacks/
│   ├── apply_audio_perturbations.py
│
├── QualityMetrics/
│   ├── quality_metrics_dir.py
│   └── visqol_evaluation.m         # Virtual Speech Quality Objective Listener
│
├── results/                     # Evaluation outputs and plots
│
├── requirements.txt             # Python dependencies
└── README.md

```

---

## 📘 Reference

If you use or reference this work, please cite:

XXXXXXXXXXXXXX


---

## ⚖️ License
MIT License — see `LICENSE` file for details.

---

## 📩 Contacts
Angela D’Angelo — Universitas Mercatorum, Rome, Italy  
📧 angela.dangelo@unimercatorum.it  



