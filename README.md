
# Joint Identification of RF Signals

## ✅ What We've Done
- **Signal Processing Pipeline**
  - Automated MATLAB scripts to generate synthetic datasets across multiple SNR levels.
- **Channel-Coding & Modulation Coverage**
  - Implemented DVB-S2 LDPC, convolutional, and turbo encoders.
  - Generated and labelled signals for 8-PSK, 32-QAM, and 64-QAM schemes.
- **Deep-Learning Models**
  - Fine-tuned ResNet-50 and custom CNNs for joint encoder-and-modulation classification (9 classes).
  - Added multi-head architectures with separate loss functions for multi-task learning.
- **Training & Evaluation**
  - Trained on 180 k samples (5 000-sample batches, 4096-bit messages) over a range of SNRs.
  - Integrated learning-rate scheduling, dropout, and PCA/t-SNE visualization for feature analysis.

## 📊 Results

### Best Training Confusion Matrix
![Training Confusion Matrix](Result/Training%20Confusion%20Matrix8680.png)

### Best Confusion Matrix for various SNRs
![Confusion Matrix for -5 SNR](Result/Confusion%20matrix%20for%20-5%20SNR.png)
![Confusion Matrix for 0 SNR](Result/Confusion%20matrix%20for%200%20SNR.png)
![Confusion Matrix for 5 SNR](Result/Confusion%20matrix%20for%205%20SNR.png)
![Confusion Matrix for 10 SNR](Result/Confusion%20matrix%20for%2010%20SNR.png)



*Models maintained stable performance across −5 dB → 14 dB SNR range with minimal degradation.*

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
