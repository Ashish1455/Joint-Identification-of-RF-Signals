import tensorflow as tf
from tensorflow.keras import layers, Model

def rnn():
    inputs = layers.Input(shape=(1024, 2), name='signal_input')
    r = layers.LSTM(64, return_sequences=True, name='lstm1')(inputs)
    r = layers.LSTM(64, name='lstm2')(r)
    r = layers.BatchNormalization()(r)
    r = layers.Activation(tf.nn.gelu)(r)
    r = layers.Dropout(0.4)(r)

    x_final = layers.Dense(units=128, activation='relu', name='dense_128')(r)
    x_final = layers.BatchNormalization(axis=1, name='bn_dense_128')(x_final)
    x_final = layers.Dropout(rate=0.3, name='dropout_dense_128')(x_final)

    # Output layer
    outputs = layers.Dense(units=12, activation='softmax', name='output_softmax')(x_final)
    model = Model(inputs=inputs, outputs=outputs)

    return model
