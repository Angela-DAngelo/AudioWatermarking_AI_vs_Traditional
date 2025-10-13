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

## 🤖 Systems Included

- **STAMP**: **S**pectral **T**ransform-domain **A**udio **M**arking with **P**erceptual model (proposed classical system)
- **AudioSeal**: https://github.com/facebookresearch/audioseal 
- **WavMark**: https://github.com/wavmark/wavmark
- **SilentCipher**: https://github.com/sony/silentcipher

---

## ⚙️ Evaluation Framework

- **Datasets:** AudioMarkBench, LibriSpeech, FMA  
- **Attacks:** Time Stretch, Gaussian Noise, Background Noise, Opus, EnCodec, Quantization, Highpass filter, Lowpass filter, Smooth, Echo, Mp3 compression 
- **Metrics:**  
  - Bit Recovery Accuracy (ACC)  
  - False Positive / Negative Rates (FPR, FNR)  
  - Audio Quality: SNR, PESQ, STOI, ViSQOL  

---

## 📁 Repository Structure

The repository is organized into three main directories:
- **Dataset/** – contains the audio datasets used for benchmarking and evaluation.
- **Systems/** – includes implementations of both classical and AI-based watermarking systems.
- **Attacks/** – provides scripts for generating and applying signal perturbations.

```text
AudioWatermarking_AI_vs_Traditional/
│
├── Dataset/
│   ├── AudioMarkBench/
│   ├── LibriSpeech/
│   └── FMA/
│
├── Systems/
│   ├── STAMP/
│   ├── AudioSeal/
│   ├── WavMark/
│   └── SilentCipher/
│
├── Attacks/
│   ├── apply_audio_perturbations.py
│
└── results/

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



