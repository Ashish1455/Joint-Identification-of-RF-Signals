from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from data_preprocessing import *
from confusin_matrix import *
from models import *

tf.keras.backend.set_floatx('float16')

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
        )
        print(f"GPU configured successfully: {len(gpus)} GPU(s) available")
    except RuntimeError as e:
        print(f"GPU configuration failed: {e}")
else:
    print("No GPUs found!")

#model, base_model = create_resnet50_model(input_shape=(4080, 2), num_classes=9)
model = LACNet(seq_len=1024, in_channels=2, base_channels=64, num_classes=12)
# model = model_2()
print(model.summary())
x_train_final, y_train_final, x_val, y_val = data_loader('Data/paper_dataset_SNR_',)
# data = np.load('./Matlab/features_spectrogram_scd_from_5_SNR.npz')
# inputs = data['spectrograms']
# label = data['labels']
# x_train_final, x_val, y_train_final, y_val =  train_test_split(inputs, label, test_size=0.2, stratify=label, random_state=42)
# print(x_train_final.shape, y_train_final.shape, x_val.shape, y_val.shape)


while True:
    train = input("Enter 'y' to train model: ")
    if train == 'y':
        # losses = {
        #     "encoder_network": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction="auto", name="encoder_loss"),
        #     "modulation_network": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction="auto", name="mod_loss"),
        #     #"interleaver_out": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction="auto", name="inter_loss"),
        # }
        # metrics = {
        #     "encoder_network": [tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
        #     "modulation_network": [tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
        #     #"interleaver_out": [tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
        # }

        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True,
            verbose=1,
            min_delta=0.001
        )

        checkpoint = ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=True,
            overwrite=True,
            verbose=1
        )

        checkpoint2 = ModelCheckpoint(
            'last_model.h5',
            save_best_only=False,
            save_weights_only=True,
            overwrite=True,
            verbose=1
        )

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])

        history = model.fit(
            x_train_final,
            # {'encoder_network' :y_train_final[0],
            #  'modulation_network' :y_train_final[1]},
            y_train_final,
            epochs=100,
            validation_data=(x_val, y_val),
            # {'encoder_network' :y_val[0],
            #  'modulation_network' :y_val[1]}),
            batch_size=32,
            callbacks=[early_stop, checkpoint, checkpoint2],
            verbose=1
        )
    else:
        break
# Plot Loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot Accuracy (or other metric)
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('his_CONV_RS_change.png')
plt.tight_layout()
plt.show()

results = model.evaluate(
    x_val,
    y_val,
    batch_size=64,
    verbose=1
)

for name, value in zip(model.metrics_names, results):
    print(f"{name}: {value}")

while True:
    k = input("'T' -> train, 'V' -> validaton, 'S' -> test or '' -> quit: ")
    if k == 'T':
        plot_confusion_matrix_only(model, x_train_final, y_train_final, 'Training')
    elif k == 'V':
        plot_confusion_matrix_only(model, x_val, y_val, 'Testing')
    elif k == 'S':
        snr = -5
        for i in range(4):
            x, y = data_preprocessing(f'./Matlab/signal_dataset_SNR_{snr}_db.npz')
            plot_confusion_matrix_only(model, x, y, f' {snr}SNR')
            snr += 5
    else:
        break

print("\n✅ Done running!\n")