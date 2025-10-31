from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from data_preprocessing import *
from confusion_matrix import *
from models.feature_net_model import *

def train():
  model = feature_net_9()
  print(model.summary())
  x_train_final, y_train_final, x_val, y_val = data_loader('drive/MyDrive/dataset/training/input_1024/dataset_Rayleigh_SNR_')

  while True:
      train = input("Enter 'y' to train model: ")
      if train == 'y':
          early_stop = EarlyStopping(
              monitor='val_accuracy',
              patience=12,
              verbose=1,
              min_delta=0.001
          )

          checkpoint = ModelCheckpoint(
              'drive/MyDrive/dataset/best_model.h5',
              monitor='val_accuracy',
              save_best_only=True,
              verbose=1
          )

          model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                        loss='SparseCategoricalCrossentropy',
                        metrics=['accuracy'])

          history = model.fit(
              x_train_final,
              y_train_final,
              epochs=75,
              validation_data=(x_val, y_val),
              batch_size=64,
              callbacks=[early_stop, checkpoint],
              verbose=1
          )
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
      else:
          break

print("\nDone training!\n")