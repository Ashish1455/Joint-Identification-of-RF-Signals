import tensorflow as tf
from tensorflow.keras import layers, Model

def cnn9(input_shape=(1024, 2, 1), num_classes=12):
    inputs = layers.Input(shape=input_shape, name='signal_input')

    x = layers.Conv2D(16, (1, 128), padding='same', use_bias=False, name='conv1')(inputs)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.LeakyReLU(alpha=0.01, name='lrelu1')(x)
    x = layers.MaxPooling2D((1, 2), padding='same', name='pool1')(x)

    x = layers.Conv2D(16, (1, 128), padding='same', use_bias=False, name='conv2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.LeakyReLU(alpha=0.01, name='lrelu2')(x)
    x = layers.MaxPooling2D((2, 1), padding='same', name='pool2')(x)

    x = layers.Conv2D(24, (1, 128), padding='same', use_bias=False, name='conv3')(x)
    x = layers.BatchNormalization(name='bn3')(x)
    x = layers.LeakyReLU(alpha=0.01, name='lrelu3')(x)
    x = layers.MaxPooling2D((2, 1), padding='same', name='pool3')(x)

    x = layers.Conv2D(24, (1, 128), padding='same', use_bias=False, name='conv4')(x)
    x = layers.BatchNormalization(name='bn4')(x)
    x = layers.LeakyReLU(alpha=0.01, name='lrelu4')(x)
    x = layers.MaxPooling2D((2, 1), padding='same', name='pool4')(x)

    x = layers.Conv2D(32, (1, 128), padding='same', use_bias=False, name='conv5')(x)
    x = layers.BatchNormalization(name='bn5')(x)
    x = layers.LeakyReLU(alpha=0.01, name='lrelu5')(x)
    x = layers.MaxPooling2D((2, 1), padding='same', name='pool5')(x)

    x = layers.Conv2D(32, (1, 128), padding='same', use_bias=False, name='conv6')(x)
    x = layers.BatchNormalization(name='bn6')(x)
    x = layers.LeakyReLU(alpha=0.01, name='lrelu6')(x)
    x = layers.MaxPooling2D((2, 1), padding='same', name='pool6')(x)

    x = layers.Conv2D(48, (1, 128), padding='same', use_bias=False, name='conv7')(x)
    x = layers.BatchNormalization(name='bn7')(x)
    x = layers.LeakyReLU(alpha=0.01, name='lrelu7')(x)
    x = layers.MaxPooling2D((2, 1), padding='same', name='pool7')(x)

    x = layers.Conv2D(48, (1, 128), padding='same', use_bias=False, name='conv8')(x)
    x = layers.BatchNormalization(name='bn8')(x)
    x = layers.LeakyReLU(alpha=0.01, name='lrelu8')(x)
    x = layers.MaxPooling2D((2, 1), padding='same', name='pool8')(x)

    x = layers.Conv2D(64, (1, 128), padding='same', use_bias=False, name='conv9')(x)
    x = layers.BatchNormalization(name='bn9')(x)
    x = layers.LeakyReLU(alpha=0.01, name='lrelu9')(x)
    x = layers.MaxPooling2D((2, 1), padding='same', name='pool9')(x)

    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(128, activation='relu', name='fc1')(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(40)(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = layers.BatchNormalization()(x)
    out = layers.Dense(num_classes, activation='softmax', name='fc2')(x)

    return Model(inputs=inputs, outputs=out, name='cnn9_new_paper_dataset')