# AudioWatermarking_AI_vs_Traditional

### Comparative Study of AI-based and Traditional Audio Watermarking Methods

This repository accompanies the research work *"Assessing Progress over a Decade of Digital Audio Watermarking Research"* (IEEE Access, 2024).

It provides a comparative framework between classical signal-processing watermarking (STAMP) and modern AI-based approaches such as AudioSeal, WavMark, and SilentCipher.

---

## 🔍 Overview

Digital audio watermarking is a core technology for copyright protection, authenticity verification, and synthetic media detection.  
While recent AI-based models have been proposed for watermark embedding and detection, their actual robustness compared to traditional designs remains uncertain.

This project evaluates both paradigms under identical conditions, assessing:
- **Imperceptibility**
- **Robustness**
- **Bit recovery accuracy**

---

## 🧩 Systems Included

- **STAMP**: **S**pectral **T**ransform-domain **A**udio **M**arking with **P**erceptual model (proposed classical system)
- **AudioSeal**: https://github.com/facebookresearch/audioseal 
- **WavMark**: https://github.com/wavmark/wavmark
- **SilentCipher**: https://github.com/sony/silentcipher

---

## 🧠 Evaluation Framework

- **Datasets:** AudioMarkBench, LibriSpeech, FMA  
- **Attacks:** Time Stretch, Gaussian Noise, Background Noise, Opus, EnCodec, Quantization, Highpass filter, Lowpass filter, Smooth, Echo, Mp3 compression 
- **Metrics:**  
  - Bit Recovery Accuracy (ACC)  
  - False Positive / Negative Rates (FPR, FNR)  
  - Audio Quality: SNR, PESQ, STOI, ViSQOL  

---

## 📚 Reference

If you use or reference this work, please cite:

@article{dangelo2025watermarking,
title={Assessing Progress over a Decade of Digital Audio Watermarking Research},
author={D'Angelo, Angela and Abrardo, Andrea and Caldelli, Roberto and Barni, Mauro},
journal={IEEE Access},
year={2025},
doi={xxxxxx}
}


---

## 🧾 License
MIT License — see `LICENSE` file for details.

---

## 📩 Contacts
Angela D’Angelo — Universitas Mercatorum, Rome, Italy  
📧 angela.dangelo@unimercatorum.it  



