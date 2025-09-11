from confusin_matrix import *
from data_preprocessing import *
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import layers, Model

#x_train, y_train, x_test, y_test, x_val, y_val = data_loader('./Data/turbo_dataset_simplified_realimag2.npz')
model = load_model("last_model.h5")
# snr = 5
# x, y = data_preprocessing(f'Matlab/signal_dataset_SNR_5_db.npz')
# plot_confusion_matrix_only(model, x, y, f'{snr}SNR')
x_train, y_train, x_val, y_val = data_loader(f'./Matlab/signal_dataset_SNR_5_db.npz')

# predictions = model.predict(x_test, batch_size=128, verbose=1)
# predictions = np.argmax(predictions, axis=1)
# train_matrix(model, x_train, y_train)
while True:
    k = input("'T' -> train, 'V' -> validaton, 'S' -> test or '' -> quit: ")
    if k == 'T':
        plot_confusion_matrix_only(model, x_train, y_train, 'Training')
    elif k == 'V':
        plot_confusion_matrix_only(model, x_val, y_val, 'Validation')
    elif k == 'S':
        snr = -5
        for i in range(4):
            x, y = data_preprocessing(f'./Matlab/signal_dataset_SNR_{snr}_db.npz')
            plot_confusion_matrix_only(model, x, y, f'{snr}SNR')
            snr += 5
    else:
        break

print("\n✅ Done running!\n")