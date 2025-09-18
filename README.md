
# Joint Identification of RF Signals

## ✅ What We've Done
- **Signal Processing Pipeline**
  - Automated MATLAB scripts to generate synthetic datasets across multiple SNR levels.
- **Channel-Coding & Modulation Coverage**
  - Implemented Turbo, Convolutional, and Polar encoders.
  - Generated and labelled signals for 8-FSK, 8-PSK, 32-QAM, and 64-QAM schemes.
- **Deep-Learning Models**
  - Fine-tuned ResNet-50 and custom CNNs Model (feature_net (106k parameter)) for joint encoder-and-modulation classification (12 classes).
  - Added multi-head architectures with separate loss functions for multi-task learning.
- **Training & Evaluation**
  - Trained on 180 k samples (500 -samples per class per SNR, 1024 IQ messages) over a range of 0dB to 10dB SNRs.
  - Integrated learning-rate scheduling, dropout, and PCA/t-SNE visualization for feature analysis.

## 📊 Results

### Training Confusion Matrix
![Training Confusion Matrix](Result/ConfusionMatrixforTraining.png)

### Confusion Matrix for various SNRs
![Confusion Matrix for -5 SNR](Result/ConfusionMatrixforTesting.png)



*Models maintained stable performance across 0 dB → 10 dB SNR range with minimal degradation.*

## 🚀 Future Expectations
1. **Interleaver Identification**  
   • Integrate interleaver patterns into the dataset.  
   • Extend the network for *triple* classification: encoder, interleaver, modulation.

2. **Channel Models**  
   • Retrain and benchmark over AWGN and Rayleigh-fading channels.

3. **Real-World Validation**  
   • Deploy with GNU Radio and SDR hardware for live over-the-air testing.
   
## Contributors

- **Ashish Tandi** — Lead Developer   
- **Aryan Jaiswal** — Lead Developer

---
*This repository is part of an ongoing Bachelor's Thesis on blind RF-signal identification.*
