import tensorflow as tf
from tensorflow.keras import layers, Model

def feature_net_9():
    inputs = layers.Input(shape=(1024, 2))
    inputs = layers.Attention()([inputs, inputs])
    
    p1 = layers.Conv1D(filters=16, kernel_size=4, padding='same')(inputs)
    p2 = layers.Conv1D(filters=16, kernel_size=8, padding='same')(inputs)
    p3 = layers.Conv1D(filters=16, kernel_size=16, padding='same')(inputs)

    x1 = layers.Concatenate()([p3, p2, p1])
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Activation('gelu')(x1)

    p4 = layers.Conv1D(filters=64, kernel_size=4, padding='same')(x1)
    p5 = layers.Conv1D(filters=64, kernel_size=8, padding='same')(x1)
    p6 = layers.Conv1D(filters=64, kernel_size=16, padding='same')(x1)

    x2 = layers.Concatenate()([p4, p6, p5])
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Activation('gelu')(x2)

    p7 = layers.Conv1D(filters=16, kernel_size=4, padding='same')(x2)
    p8 = layers.Conv1D(filters=16, kernel_size=8, padding='same')(x2)
    p9 = layers.Conv1D(filters=16, kernel_size=16, padding='same')(x2)

    x3 = layers.Concatenate()([p8, p7, p9])
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.Activation('gelu')(x3)
    x3 = layers.Add()([x3, x1])
    x = layers.Attention()([x3, x3])

    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(128, activation='gelu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    output = layers.Dense(12, activation='softmax')(x)

    return Model(inputs=inputs, outputs=output)