from tensorflow.keras import regularizers
import tensorflow as tf
from tensorflow.keras import layers, Model

def res_block(x, filters, kernel_size=3, stride=1):
    shortcut = x
    x = layers.Conv1D(filters, kernel_size, strides=stride, padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(filters, kernel_size, padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, strides=stride, padding='same', kernel_regularizer=regularizers.l2(1e-4))(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def ResNet1D():
    inputs = layers.Input(shape=(1024, 2))
    x = layers.Conv1D(32, kernel_size=7, strides=2, padding='same', kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.1)(x)

    x = res_block(x, 32, kernel_size=3, stride=1)
    x = res_block(x, 32, kernel_size=3, stride=1)
    x = res_block(x, 256, kernel_size=3, stride=2)  
    x = res_block(x, 256, kernel_size=3, stride=1)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-3))(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(12, activation='softmax')(x) 

    return Model(inputs, outputs)