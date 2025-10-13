## Datasets

This study evaluates audio watermarking systems across three heterogeneous datasets.

---

### 🔹 [AudioMark](https://github.com/moyangkuo/AudioMarkBench)
**AudioMark**  refers to **AudioMarkBench**, a benchmark introduced by Liu *et al.* (*NeurIPS Datasets and Benchmarks Track*, 2024).  
We used the **AudioMarkData** subset, derived from *Mozilla Common Voice*, which includes balanced coverage across **25 languages**, **two genders (male/female)**, and **four age groups** (teens, twenties, thirties, forties).  
Each sample is up to **5 seconds** long at **16 kHz**, and we selected **200 clips** uniformly distributed across all attributes.

---

### 🔹 [LibriSpeech](https://www.openslr.org/12)
**LibriSpeech** (Panayotov *et al.*, *ICASSP 2015*) is a large-scale corpus of English read speech derived from public domain audiobooks.  
It contains over **1,000 hours** of recordings at **16 kHz**, covering a broad range of speakers and speaking styles.  
For this study, we sampled **200 clips** (maximum 5 seconds each) to represent natural English speech.

---

### 🔹 [Free Music Archive (FMA)](https://github.com/mdeff/fma)
The **Free Music Archive (FMA)** dataset (Defferrard *et al.*, *ISMIR 2017*) is an open collection for **music analysis**, containing thousands of tracks across genres such as classical, jazz, pop, and electronic.  
Musical content introduces unique challenges for watermarking due to its **broader spectral dynamics and complexity**.  
Following the same protocol, we randomly selected **200 tracks**, each clipped to **5 seconds**.
