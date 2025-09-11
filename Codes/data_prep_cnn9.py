import numpy as np
import tensorflow as tf
import h5py
import scipy.io as sio
import mat73

def load_dataset(mat_path: str):
    def _post(X, codeIdx, modIdx, snrIdx):
        x = np.asarray(X, dtype=np.float32)  # (N,1024,2)
        x = np.expand_dims(x, axis=-1)  # (N,1024,2,1)
        x = np.transpose(x, (0, 2, 1, 3))  # (N,2,1024,1)
        x_mean = np.mean(x, axis=2, keepdims=True)  # (N,2,1,1)
        x = x - x_mean
        rms = np.sqrt(np.mean(np.square(x), axis=(1, 2, 3), keepdims=True) + 1e-8)  # (N,1,1,1)
        x = x / rms
        codeIdx = np.asarray(codeIdx).astype(np.int32).ravel()
        modIdx = np.asarray(modIdx).astype(np.int32).ravel()
        snrIdx = np.asarray(snrIdx).astype(np.int32).ravel()
        C = int(np.max(codeIdx))
        M = int(np.max(modIdx))
        S = int(np.max(snrIdx))
        y = (codeIdx - 1) * M + (modIdx - 1)
        y = y.astype(np.int32)
        return x, y, (C, M, S)

    def _fix_X_shape(X):
        X = np.array(X)
        if X.ndim != 3:
            raise ValueError('X must be 3D')
        # Prefer (N, 1024, 2)
        if X.shape[-1] == 2 and X.shape[-2] == 1024:
            return X
        # (2, 1024, N) -> (N, 1024, 2)
        if X.shape[0] == 2 and X.shape[1] == 1024:
            return np.transpose(X, (2, 1, 0))
        # (1024, 2, N) -> (N, 1024, 2)
        if X.shape[0] == 1024 and X.shape[1] == 2:
            return np.transpose(X, (2, 0, 1))
        # (N, 2, 1024) -> (N, 1024, 2)
        if X.shape[1] == 2 and X.shape[2] == 1024:
            return np.transpose(X, (0, 2, 1))
        raise ValueError(f'Unexpected X shape {X.shape}, expected last two dims (1024,2)')

    if h5py is not None:
        try:
            with h5py.File(mat_path, 'r') as f:
                X = _fix_X_shape(f['X'])
                codeIdx = np.array(f['codeIdx'])
                modIdx = np.array(f['modIdx'])
                snrIdx = np.array(f['snrIdx'])
                return _post(X, codeIdx, modIdx, snrIdx)
        except Exception:
            pass

def make_datasets(x, y, val_split=0.1, test_split=0.1, shuffle=True, seed=42, batch_size=256):
    N = x.shape[0]
    if val_split + test_split >= 1.0:
        raise ValueError('val_split + test_split must be < 1.0')
    idx = np.arange(N)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    x = x[idx]
    y = y[idx]

    n_val = int(N * val_split)
    n_test = int(N * test_split)
    x_val, y_val = x[:n_val], y[:n_val]
    x_test, y_test = x[n_val:n_val + n_test], y[n_val:n_val + n_test]
    x_tr, y_tr = x[n_val + n_test:], y[n_val + n_test:]

    buffer = x_tr.shape[0] if shuffle else batch_size
    ds_tr = (
        tf.data.Dataset.from_tensor_slices((x_tr, y_tr))
        .shuffle(buffer, seed=seed, reshuffle_each_iteration=True)
        .batch(batch_size)
        .prefetch(2)
    )
    ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size).prefetch(2)
    ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size).prefetch(2)
    return ds_tr, ds_val, ds_test
