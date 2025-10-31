from confusion_matrix import *
from data_preprocessing import *
from models.feature_net_model import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt
import time

# Load model
model = feature_net_9()
model.load_weights("drive/MyDrive/dataset/best_model.h5")

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='SparseCategoricalCrossentropy',
              metrics=['accuracy'])
snr_values = range(-20, 30, 5)
accuracies = []

for snr in snr_values:
    # Load data
    x, y = mat_file_preprocessing(f'drive/MyDrive/dataset/testing/input_1024/dataset_Rayleigh_SNR_{snr}_dB.mat')

    # Evaluate
    loss, acc = model.evaluate(x, y, verbose=1)
    accuracies.append(acc)
    _ = model.predict(x[:10], verbose=0)
    start_time = time.time()
    _ = model.predict(x, verbose=0)
    end_time = time.time()

    total_time = end_time - start_time
    time_per_sample = (total_time / len(x)) * 1000

    # Plot confusion matrix
    plot_confusion_matrix_only(model, x, y, f'{snr}SNR')
    print(time_per_sample)
    print(f"SNR {snr} dB → Accuracy: {acc:.4f}")

# Plot accuracy vs SNR
plt.figure(figsize=(8, 5))
plt.plot(snr_values, accuracies, marker='o', linestyle='-', linewidth=2, markersize=6)
plt.title("Classification Accuracy vs SNR", fontsize=14)
plt.xlabel("SNR (dB)", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.xticks(snr_values)
plt.ylim([0.4, 1.05])
plt.savefig("Accuracy_vs_SNR.png", dpi=300)
plt.show()

print("\nDone Testing!\n")