# Joint Identification of RF Signals

Blind, joint classification of **channel-coding scheme** and **modulation format** for RF signals, using deep learning on synthetic IQ datasets. Part of an ongoing Bachelor's Thesis on blind RF-signal identification.

## 📁 Repository Structure

| Folder / File | Contents |
|---|---|
| `Codes/` | Python training, evaluation, and pre-processing scripts for the deep-learning models |
| `Matlab Codes/` | MATLAB scripts for synthetic dataset generation (encoders, modulators, SNR sweep) |
| `GNU Radio Code/` | GNU Radio flowgraphs / scripts for real-world signal capture and inference, first step toward SDR deployment |
| `models/` | Model architecture definitions (ResNet-50 fine-tune, custom `feature_net`, multi-head classifier) |
| `Saved Models/` | Pre-trained model checkpoints for direct inference/evaluation |
| `Result/` | Confusion matrices, PCA/t-SNE plots, and additional per-SNR result figures |
| `requirements.txt` | Python dependencies needed to run the codebase |

## ✅ What We've Done

- **Signal Processing Pipeline**
  - Automated MATLAB scripts to generate synthetic datasets across multiple SNR levels.
- **Channel-Coding & Modulation Coverage**
  - Implemented Turbo, Convolutional, and Polar encoders.
  - Generated and labelled signals for 8-FSK, 8-PSK, 32-QAM, and 64-QAM schemes.
- **Deep-Learning Models**
  - Fine-tuned ResNet-50 and a custom CNN (`feature_net`, ~106k parameters) for joint encoder-and-modulation classification (12 classes).
  - Added multi-head architectures with separate loss functions for multi-task learning.
- **Training & Evaluation**
  - Trained on 66k samples (500 samples per class per SNR, 1024 IQ symbols per message) over a 0 dB to 10 dB SNR range.
  - Integrated learning-rate scheduling, dropout, and PCA/t-SNE visualization for feature analysis.
- **Real-World Groundwork**
  - Added initial GNU Radio flowgraphs/scripts for interfacing trained models with SDR hardware, the first step toward the live over-the-air validation goal below.

## 📊 Results

### Training Confusion Matrix
![Training Confusion Matrix](Result/Confusion%20Matrix%20for%20Training.png)

### Confusion Matrix for Various SNRs
![Confusion Matrix for Testing](Result/Confusion%20Matrix%20for%20Testing.png)

*Models maintained stable performance across the 0 dB → 10 dB SNR range with minimal degradation.*
*See the `Result/` folder for additional figures.*

## ⚙️ Getting Started

```bash
git clone https://github.com/Ashish1455/Joint-Identification-of-RF-Signals.git
cd Joint-Identification-of-RF-Signals
pip install -r requirements.txt
```

- Use the scripts in `Matlab Codes/` to (re)generate the synthetic dataset.
- Use the scripts in `Codes/` to train new models, or load a checkpoint from `Saved Models/` to run evaluation directly.
- Use `GNU Radio Code/` to try inference on live/SDR-captured signals.

## 🚀 Future Expectations

1. **Interleaver Identification**
   - Integrate interleaver patterns into the dataset.
   - Extend the network for *triple* classification: encoder, interleaver, modulation.
2. **Channel Models**
   - Retrain and benchmark over AWGN and Rayleigh-fading channels.
3. **Real-World Validation**
   - Build on the initial GNU Radio integration to complete full over-the-air testing with SDR hardware.

## Contributors

- **Ashish Tandi** — Lead Developer
- **Aryan Jaiswal** — Lead Developer

---

*This repository is part of an ongoing Bachelor's Thesis on blind RF-signal identification.*
