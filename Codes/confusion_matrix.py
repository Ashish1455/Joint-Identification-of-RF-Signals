
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

def plot_confusion_matrix_only(model, x_train, y_train, title, scale=1):
    class_names = ['BPSK + TURBO', 'BPSK + CONV', 'BPSK + POLAR', '8PSK + TURBO', '8PSK + CONV', '8PSK + POLAR', 
                   '16QAM + TURBO', '16QAM + CONV', '16QAM + POLAR', '64QAM + TURBO', '64QAM + CONV', '64QAM + POLAR']

    if scale == 1:
        scale = max(1, int(len(y_train) / 1200)) 
    else:
        scale = max(1, int(scale))

    print(f"🔄 Processing {len(y_train)} samples with scale factor: {scale}")

    try:
        if tf.config.list_physical_devices('GPU'):
            try:
                with tf.device('/GPU:0'):
                    tf.keras.backend.clear_session()
                    y_pred = model.predict(x_train, batch_size=32, verbose=1)
            except Exception as gpu_error:
                print(f"GPU prediction failed: {gpu_error}")
                print("Falling back to CPU...")
                with tf.device('/CPU:0'):
                    y_pred = model.predict(x_train, batch_size=64, verbose=1)
        else:
            print("No GPU available, using CPU...")
            with tf.device('/CPU:0'):
                y_pred = model.predict(x_train, batch_size=64, verbose=1)

    except Exception as e:
        raise RuntimeError(f"Model prediction failed: {e}")

    y_train = y_train.astype(int)
    y_pred = y_pred.astype(int)

    all_classes = np.union1d(np.unique(y_train), np.unique(y_pred))
    n_classes = len(all_classes)
    print(f"📋 Found {n_classes} classes: {all_classes}")

    # Create mapping for consistent indexing
    class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}
    y_train_mapped = np.array([class_to_idx[cls] for cls in y_train])
    y_pred_mapped = np.array([class_to_idx[cls] for cls in y_pred])

    # Generate confusion matrix
    cm = confusion_matrix(y_train_mapped, y_pred_mapped)

    # Convert confusion matrix to percentages
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Handle division by zero (if any class has no samples)
    cm_percentage = np.nan_to_num(cm_percentage, nan=0.0)

    # Apply scaling to original matrix for display purposes if needed
    if scale > 1:
        scaled_cm = cm // scale
        scaled_cm = np.maximum(scaled_cm, (cm > 0).astype(int))
        scaled_cm_percentage = scaled_cm.astype('float') / scaled_cm.sum(axis=1)[:, np.newaxis] * 100
        scaled_cm_percentage = np.nan_to_num(scaled_cm_percentage, nan=0.0)
    else:
        scaled_cm = cm
        scaled_cm_percentage = cm_percentage

    # Ensure confusion matrix has correct dimensions
    if scaled_cm_percentage.shape[0] != n_classes or scaled_cm_percentage.shape[1] != n_classes:
        new_cm = np.zeros((n_classes, n_classes), dtype=float)
        rows = min(scaled_cm_percentage.shape[0], n_classes)
        cols = min(scaled_cm_percentage.shape[1], n_classes)
        new_cm[:rows, :cols] = scaled_cm_percentage[:rows, :cols]
        scaled_cm_percentage = new_cm

    # Calculate accuracy
    accuracy = np.mean(y_train == y_pred) * 100
    print(f"📊 Overall Accuracy: {accuracy:.2f}%")

    plt.figure(figsize=(12, 10))

    # Create custom annotations with percentage signs
    annot_array = np.array([[f"{val:.1f}%" for val in row] for row in scaled_cm_percentage])

    # Create class labels using short names
    pred_labels = []
    true_labels = []
    for cls in all_classes:
        if cls < len(class_names):
            pred_labels.append(class_names[int(cls)])
            true_labels.append(class_names[int(cls)])
        else:
            pred_labels.append(f'Class_{int(cls)}')
            true_labels.append(f'Class_{int(cls)}')

    # heatmap with percentage values
    mask = scaled_cm_percentage == 0
    sns.heatmap(scaled_cm_percentage,
                annot=annot_array,
                fmt='',
                cmap='Blues',
                xticklabels=pred_labels,
                yticklabels=true_labels,
                cbar_kws={'label': 'Percentage (%)'},
                square=True,
                linewidths=0.5,
                linecolor='white',
                mask=mask if np.any(mask) else None,
                vmin=0, vmax=100)

    plt.xlabel('Predicted Classes', fontsize=24, fontweight='bold')
    plt.ylabel('True Classes', fontsize=24, fontweight='bold')
    plt.title(f'Accuracy: {accuracy:.2f}%', fontsize=22)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tick_params(axis='x', labelsize=16)
    plt.tick_params(axis='y', labelsize=16)

    plt.text(0.5, 1.05, title,
             fontsize=20, fontweight='bold', ha='center', va='bottom',
             transform=plt.gca().transAxes)

    save_dir = './confusion_matrices/'
    os.makedirs(save_dir, exist_ok=True)

    clean_title = ''.join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
    filename = f'Confusion_Matrix_{clean_title}.png'
    filepath = os.path.join(save_dir, filename)

    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"💾 Saved plot to: {filepath}")

    plt.tight_layout()
    plt.show()