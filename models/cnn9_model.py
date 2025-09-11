import tensorflow as tf
from tensorflow.keras import layers, Model

# CNN-9 per diagram, with I/Q fusion in the first conv layer.
def build_cnn9(input_shape=(2, 1024, 1), num_classes=20):

    inp = layers.Input(shape=input_shape, name="iq_input")

    def block(x, filters, idx):
        if idx == 1:
            x = layers.Conv2D(
                filters,
                (2, 128),
                padding="valid",
                use_bias=False,
                name=f"conv{idx}",
            )(x)
        else:
            x = layers.Conv2D(
                filters,
                (1, 128),
                padding="same",
                use_bias=False,
                name=f"conv{idx}",
            )(x)
        x = layers.BatchNormalization(name=f"bn{idx}")(x)
        x = layers.LeakyReLU(alpha=0.1, name=f"lrelu{idx}")(x)
        x = layers.MaxPooling2D(pool_size=(1, 2), padding="same", name=f"pool{idx}")(x)
        return x

    filters_list = [16, 16, 24, 24, 32, 32, 48, 48, 64]
    x = inp
    for i, f in enumerate(filters_list, start=1):
        x = block(x, f, i)

    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(128, activation="relu", name="fc1")(x)
    out = layers.Dense(num_classes, activation="softmax", name="fc2")(x)

    return Model(inp, out, name="cnn9_exact")

