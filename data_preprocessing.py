import numpy as np
import h5py
import librosa
from sklearn.model_selection import train_test_split


def decode_class_id(class_id, num_encoders=3, num_modulations=3):
    encoder_id, mod_id = [], []
    for i in class_id:
        total_per_encoder = num_modulations
        encoder_id.append(i // total_per_encoder)
        mod_id.append(i % total_per_encoder)
    return np.array([encoder_id, mod_id])


def data_preprocessing(path):
    """Load a single v7.3 .mat file (HDF5 format)."""
    with h5py.File(path, "r") as f:
        # Adjust dataset keys based on your .mat structure
        real = np.array(f['signals_real']).T      # h5py loads in column-major order
        imag = np.array(f['signals_imag']).T
        label = np.array(f['labels_class']).squeeze()

    inputs = np.stack((real, imag), axis=-1).astype(np.float16)
    return inputs, label


def data_loader(base_path, num_files=10):
    """Load multiple .mat files and combine them into one dataset."""
    all_inputs, all_labels = [], []

    for i in range(num_files):
        inputs, labels = data_preprocessing(f"{base_path}{i}_db.mat")
        all_inputs.append(inputs)
        all_labels.append(labels)

    # Concatenate across files
    inputs = np.concatenate(all_inputs, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    print("Combined input shape:", inputs.shape)
    print("Combined label shape:", labels.shape)

    # Train/val split
    x_train, x_val, y_train, y_val = train_test_split(
        inputs, labels,
        test_size=0.2,
        stratify=labels,
        random_state=42
    )

    return x_train, y_train, x_val, y_val


def extract_features(signal, sr=16000, n_fft=512, hop_length=256):
    specs, feats = [], []
    for ch in range(signal.shape[1]):
        y = signal[:, ch]

        # Spectrogram (magnitude)
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        specs.append(S)

        # Spectral Contrast Descriptor (mean across time)
        scd = librosa.feature.spectral_contrast(S=S, sr=sr, n_fft=n_fft, hop_length=hop_length)
        feats.append(np.mean(scd, axis=1))

    spectrograms = np.stack(specs, axis=0)           # (2, F, T)
    scd_features = np.concatenate(feats, axis=0)    # (14,)
    return spectrograms, scd_features