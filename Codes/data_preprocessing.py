import numpy as np
import h5py
from sklearn.model_selection import train_test_split

def mat_file_preprocessing(path):
    with h5py.File(path, "r") as f:
        real = np.array(f['signals_real']).T
        imag = np.array(f['signals_imag']).T
        label = np.array(f['labels_class']).squeeze()

    inputs = np.stack((real, imag), axis=-1).astype(np.float16)
    return inputs, label

def npz_file_preprocessing(path):
    data = np.load(path)
    real = data['signals_real']
    imag = data['signals_imag']
    label = data['labels_class'].transpose()

    inputs = np.stack((real, imag), axis=-1).astype(np.float16)
    return inputs, label

def data_loader(base_path):
    """Load multiple .mat files and combine them into one dataset."""
    all_inputs, all_labels = [], []

    for i in range(-20, 30, 5):
        inputs, labels = mat_file_preprocessing(f"{base_path}{i}_dB.mat")
        all_inputs.append(inputs)
        all_labels.append(labels)

    inputs = np.concatenate(all_inputs, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    print("Combined input shape:", inputs.shape)
    print("Combined label shape:", labels.shape)
    print('classes', np.unique(labels))
    print('Count of label 0:', np.sum(labels == 0))

    x_train, x_val, y_train, y_val = train_test_split(inputs, labels, test_size=0.2, stratify=labels, random_state=42)
    return x_train, y_train, x_val, y_val