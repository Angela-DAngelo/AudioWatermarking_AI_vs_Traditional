### Audio Quality Metrics

This section describes the **audio quality metrics** used to evaluate the effect of watermarking on audio/speech signals.

| **Metric** | **Description** | **Typical Range** | **Interpretation** |
|-------------|-----------------|------------------:|--------------------|
| **MSE (Mean Squared Error)** | Measures the average squared difference between the original and processed signals. Lower values indicate higher similarity. | 0 – 0.01 | Lower is better |
| **SNR (Signal-to-Noise Ratio)** | Measures the ratio between the original signal power and the noise introduced by processing. Higher values indicate better preservation of the signal. | 0 – 40 dB | Higher is better |
| **SI-SNR (Scale-Invariant Signal-to-Noise Ratio)** | Similar to SNR but invariant to scale differences between signals, often used in speech enhancement tasks. | 0 – 40 dB | Higher is better |
| **PESQ (Perceptual Evaluation of Speech Quality)** | Standard ITU-T P.862 perceptual metric for speech quality. Outputs a Mean Opinion Score (MOS) prediction. | 1 – 4.5 | Higher is better |
| **STOI (Short-Time Objective Intelligibility)** | Estimates speech intelligibility by comparing short-time temporal envelopes between reference and degraded signals. | 0 – 1 | Higher is better |
| **ViSQOL (Virtual Speech Quality Objective Listener)** | Full-reference perceptual metric developed by Google, estimating MOS-LQO (Mean Opinion Score – Listening Quality Objective). | 1 – 5 | Higher is better |

---

### ViSQOL Implementation

The **ViSQOL** metric was computed using the official MATLAB implementation available from MathWorks:  
🔗 [https://it.mathworks.com/help/audio/ref/visqol.html](https://it.mathworks.com/help/audio/ref/visqol.html)

---

### MSE,SI-SNR, SNR, PESQ and STOI implementation

The metrics **MSE**, **SI-SNR**, **PESQ**, and **STOI** are implemented in the Python script [`quality_metrics.py`](./quality_metrics.py):contentReference[oaicite:0]{index=0}.  
This script automatically compares pairs of audio files (original vs. watermarked) and saves the results in an Excel file.

### Usage

From the command line:
```bash
python quality_metrics.py --orig path/to/originals 
                          --marchiati path/to/watermarked 
                          --sr 16000 
                          --out results.xlsx

```

### Arguments

| **Argument** | **Description** | **Default** |
|---------------|-----------------|-------------|
| `--orig` | Path to the folder containing **original** audio files (`.wav`) | — |
| `--marchiati` | Path to the folder containing **watermarked** audio files (`.wav`) | — |
| `--sr` | Target sample rate used when loading audio (in Hz) | `16000` |
| `--out` | Output Excel filename where results will be saved | `results.xlsx` |

🔹 Watermarked files are automatically matched by their **numeric ID** in the filename  
(e.g. `0001.wav` is matched with `wm_0001.wav`).

---

### Output

After execution, the script prints per-file results and saves a summary Excel file with all metrics:

| **file** | **mse** | **snr** | **si-snr** | **stoi** | **pesq** |
|-----------|----------|----------|-------------|-----------|-----------|
| 001 | 0.0015 | 23.45 | 22.87 | 0.9851 | 4.20 |
| 002 | 0.0020 | 20.75 | 19.92 | 0.9723 | 3.85 |
| ... | ... | ... | ... | ... | ... |

Results are saved to the specified output file (i.e.: `results.xlsx`).
The console also displays average values of all metrics at the end of processing:

---

### Notes

- Input files must be mono `.wav` audio files sampled at the same rate (default 16 kHz).  
- PESQ, and STOI are implemented in Python using the official libraries:
  - [`pesq`](https://pypi.org/project/pesq/)
  - [`pystoi`](https://pypi.org/project/pystoi/)
- PESQ in wideband mode (`'wb'`) may issue warnings for short signals (<3 s).  
- PESQ and STOI assume time alignment between the reference and processed files.  




