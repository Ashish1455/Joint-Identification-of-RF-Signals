from sklearn.model_selection import train_test_split
import numpy as np
import librosa
from tensorflow.keras.utils import to_categorical

def decode_class_id(class_id, num_encoders=3, num_modulations=3):
    encoder_id, mod_id = [], []
    for i in class_id:
        total_per_encoder = num_modulations
        encoder_id.append(i // total_per_encoder)
        mod_id.append(i % total_per_encoder)

    return np.array([encoder_id, mod_id])

def data_preprocessing(path):
    data = np.load(path)
    # real = data['real_parts']  # shape: (samples, 4080)
    # imag = data['imag_parts']  # shape: (samples, 4080)
    # label = data['labels']
    real = data['signals_real']
    imag = data['signals_imag']
    label = data['labels_class']
    data.close()
    # real = np.transpose(real)
    # imag = np.transpose(imag)
    inputs = np.stack((real, imag), axis=-1).astype(np.float16)
    label = np.transpose(label)
    return inputs, label

# loading the data
def data_loader(path):
    inputs, label = data_preprocessing(path)
    # data splitting
    print(inputs.shape)
    print(label.shape)
    # validation set creating
    x_train_final, x_val, y_train_final, y_val = train_test_split(inputs, label, test_size=0.2, stratify=label, random_state=42)

    # y_val = to_categorical(y_val, num_classes=9)
    # y_train_final = to_categorical(y_train_final, num_classes=9)
    # y_val = decode_class_id(y_val)
    # y_train_final = decode_class_id(y_train_final)

    return x_train_final, y_train_final, x_val, y_val

def extract_features(signal, sr=16000, n_fft=512, hop_length=256):
    specs = []
    feats = []

    for ch in range(signal.shape[1]):
        y = signal[:, ch]

        # Spectrogram (magnitude)
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        specs.append(S)

        # Spectral Contrast Descriptor (mean across time)
        scd = librosa.feature.spectral_contrast(S=S, sr=sr, n_fft=n_fft, hop_length=hop_length)
        feats.append(np.mean(scd, axis=1))

    spectrograms = np.stack(specs, axis=0)           # (2, F, T)
    scd_features = np.concatenate(feats, axis=0)    # (14,) if 7 bands per channel

    return spectrograms, scd_features

# X_raw, y = data_preprocessing('./Matlab/signal_dataset_SNR_5_dB.npz')
# spec_list, scd_list = [], []
#
# for sig in X_raw:
#
#     spec, scd = extract_features(sig, sr=16000)
#     spec_list.append(spec)
#     scd_list.append(scd)
#
# # Convert to arrays
# spectrograms = np.array(spec_list, dtype=np.float32)   # (N, 2, F, T)
# scd_features = np.array(scd_list, dtype=np.float32)   # (N, 14)
#
# # Save to compressed file
# np.savez_compressed(
#     "./Matlab/features_spectrogram_scd_from_5_SNR.npz",
#     spectrograms=spectrograms,
#     scd_features=scd_features,
#     labels=y
# )
#
# print("Saved to features_spectrogram_scd_from_raw.npz")