import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np


def plot_confusion_matrix_only(model, x_train, y_train, title, scale=1):
    """
    Plot confusion matrix with forced CPU usage to avoid GPU memory issues
    """

    try:
        # Try GPU
        with tf.device('/GPU:0'):
            print("Trying prediction on GPU...")
            y_pred = model.predict(x_train, batch_size=32, verbose=1)
    except Exception as e:
        print(f"GPU prediction failed: {e}")
        print("Falling back to CPU...")
        with tf.device('/CPU:0'):
            y_pred = model.predict(x_train, batch_size=64, verbose=1)

    # Convert predictions to class labels if model outputs probabilities
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)

    # Rest of your existing code...
    if len(y_train.shape) > 1:
        if y_train.shape[1] == 1:
            y_train = y_train.flatten()
        else:
            raise ValueError('y_train must be 1D or single-column 2D array')

    if len(y_pred.shape) > 1:
        if y_pred.shape[1] == 1:
            y_pred = y_pred.flatten()

    y_train = y_train.astype(int)
    y_pred = y_pred.astype(int)

    all_classes = np.union1d(np.unique(y_train), np.unique(y_pred))
    n_classes = len(all_classes)

    class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}
    y_train_mapped = np.array([class_to_idx[cls] for cls in y_train])
    y_pred_mapped = np.array([class_to_idx[cls] for cls in y_pred])

    cm = confusion_matrix(y_train_mapped, y_pred_mapped)
    scaled_cm = cm // scale

    if scaled_cm.shape[0] != n_classes or scaled_cm.shape[1] != n_classes:
        new_cm = np.zeros((n_classes, n_classes), dtype=int)
        rows = min(scaled_cm.shape[0], n_classes)
        cols = min(scaled_cm.shape[1], n_classes)
        new_cm[:rows, :cols] = scaled_cm[:rows, :cols]
        scaled_cm = new_cm

    plt.figure(figsize=(12, 10))
    sns.heatmap(scaled_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f'Pred {int(cls)}' for cls in all_classes],
                yticklabels=[f'True {int(cls)}' for cls in all_classes],
                cbar_kws={'label': f'Number of Samples (÷{scale})'},
                square=True, linewidths=0.5)

    plt.xlabel('Predicted Classes', fontsize=14, fontweight='bold')
    plt.ylabel('True Classes', fontsize=14, fontweight='bold')
    plt.title(f'Confusion Matrix Heatmap {title}', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    accuracy = np.mean(y_train == y_pred) * 100
    plt.suptitle(f'Overall Accuracy: {accuracy:.2f}%',
                 fontsize=10, y=0.98)
    plt.savefig(f'Confusion Matrix for {title}.png')
    plt.tight_layout()
    plt.show()


def plot_confusion_matrices_multitask(model, x, y_dict, scale=1):
    """
    Plot confusion matrices for multitask model with outputs:
      - encoder_out
      - modulation_out
      - interleaver_out

    Tries GPU first, falls back to CPU if OOM or GPU unavailable.
    """
    tf.keras.backend.set_floatx('float16')
    try:
        # Try GPU
        with tf.device('/GPU:0'):
            print("Trying prediction on GPU...")
            y_pred_list = model.predict(x, batch_size=4, verbose=1)
    except Exception as e:
        print(f"GPU prediction failed: {e}")
        print("Falling back to CPU...")
        with tf.device('/CPU:0'):
            y_pred_list = model.predict(x, batch_size=256, verbose=1)

    # If model has dict outputs, ensure same order
    output_names = ["encoder_out", "modulation_out", "interleaver_out"]

    for i, out_name in enumerate(output_names):
        print(f"\n--- Confusion Matrix for {out_name} ---")
        y_true = y_dict[:, i]
        y_pred = y_pred_list[i]

        # Convert softmax → class indices
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            y_pred = np.argmax(y_pred, axis=1)

        if len(y_true.shape) > 1 and y_true.shape[1] == 1:
            y_true = y_true.flatten()

        y_true = y_true.astype(int)
        y_pred = y_pred.astype(int)

        # Handle mismatched classes gracefully
        all_classes = np.union1d(np.unique(y_true), np.unique(y_pred))
        n_classes = len(all_classes)

        class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}
        y_true_mapped = np.array([class_to_idx[cls] for cls in y_true])
        y_pred_mapped = np.array([class_to_idx[cls] for cls in y_pred])

        cm = confusion_matrix(y_true_mapped, y_pred_mapped)
        scaled_cm = cm // scale

        # Pad if mismatch
        if scaled_cm.shape != (n_classes, n_classes):
            new_cm = np.zeros((n_classes, n_classes), dtype=int)
            rows = min(scaled_cm.shape[0], n_classes)
            cols = min(scaled_cm.shape[1], n_classes)
            new_cm[:rows, :cols] = scaled_cm[:rows, :cols]
            scaled_cm = new_cm

        # Plot heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(scaled_cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[f'Pred {cls}' for cls in all_classes],
                    yticklabels=[f'True {cls}' for cls in all_classes],
                    cbar_kws={'label': f'Count (÷{scale})'},
                    square=True, linewidths=0.5)

        plt.xlabel('Predicted', fontsize=12, fontweight='bold')
        plt.ylabel('True', fontsize=12, fontweight='bold')
        plt.title(f'{out_name} Confusion Matrix', fontsize=14, fontweight='bold')

        acc = np.mean(y_true == y_pred) * 100
        plt.suptitle(f'Accuracy: {acc:.2f}%', fontsize=10, y=0.98)

        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{out_name}.png')
        plt.show()
