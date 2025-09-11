from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50
import tensorflow as tf
from tensorflow.keras import layers, Model

from tensorflow.keras import layers, Model
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout, BatchNormalization,
    Input, Conv2D, MaxPooling2D, Activation
)
from tensorflow.keras.regularizers import l1_l2

def model_2():
    # Input layer
    input_layer = layers.Input(shape=(4080, 2, 1), name='signal_input')

    # First convolutional block
    x = layers.Conv2D(64, (7, 7), strides=(1, 1), padding='same')(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (7, 7), strides=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (7, 7), strides=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (7, 7), strides=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # Additional Conv + BatchNorm + Activation + Dropout blocks
    x = layers.Conv2D(96, (3, 4), strides=(1, 1), padding='same' )(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.3)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(64, (7, 3), strides=(1, 1), padding='same' )(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.3)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(48, (12, 1), strides=(1, 1), padding='same' )(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.3)(x)
    x = layers.Dropout(0.3)(x)

    # Max pooling
    x = layers.AveragePooling2D(pool_size=(4, 2), padding='same')(x)

    # Flatten and Dense layers
    x = layers.Flatten()(x)

    x = layers.Dense(100, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    # Output layer with 4 classes (Softmax)
    output_layer = layers.Dense(9, activation='softmax')(x)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

def model1():
    # Input layer
    inputs = Input(shape=(2, 4080, 1), name='input')

    # Initial convolution block
    x = Conv2D(64, (1, 50), strides=(2, 2), padding='same', name='initial_conv')(inputs)
    x = BatchNormalization(name='initial_bn')(x)
    x = Activation('relu', name='initial_relu')(x)
    x = MaxPooling2D((1, 3), strides=(2, 2), padding='same', name='initial_pool')(x)

    # Block 1: Feature extraction
    x = Conv2D(64, (1, 48), padding='same', name='block1_conv1')(x)
    x = BatchNormalization(name='block1_bn1')(x)
    x = Activation('relu', name='block1_relu1')(x)

    x = Conv2D(64, (1, 46), padding='same', name='block1_conv2')(x)
    x = BatchNormalization(name='block1_bn2')(x)
    x = Activation('relu', name='block1_relu2')(x)

    x = MaxPooling2D((1, 2), name='block1_pool')(x)
    x = Dropout(0.25, name='block1_dropout')(x)

    # Block 2: Deeper feature extraction
    x = Conv2D(128, (1, 46), padding='same', name='block2_conv1')(x)
    x = BatchNormalization(name='block2_bn1')(x)
    x = Activation('relu', name='block2_relu1')(x)

    x = Conv2D(128, (1, 44), padding='same', name='block2_conv2')(x)
    x = BatchNormalization(name='block2_bn2')(x)
    x = Activation('relu', name='block2_relu2')(x)

    x = Conv2D(128, (1, 42), padding='same', name='block2_conv3')(x)
    x = BatchNormalization(name='block2_bn3')(x)
    x = Activation('relu', name='block2_relu3')(x)

    x = MaxPooling2D((1, 2), name='block2_pool')(x)
    x = Dropout(0.25, name='block2_dropout')(x)

    # Block 3: High-level features
    x = Conv2D(256, (1, 40), padding='same', name='block3_conv1')(x)
    x = BatchNormalization(name='block3_bn1')(x)
    x = Activation('relu', name='block3_relu1')(x)

    x = Conv2D(256, (1, 38), padding='same', name='block3_conv2')(x)
    x = BatchNormalization(name='block3_bn2')(x)
    x = Activation('relu', name='block3_relu2')(x)

    x = Conv2D(256, (1, 36), padding='same', name='block3_conv3')(x)
    x = BatchNormalization(name='block3_bn3')(x)
    x = Activation('relu', name='block3_relu3')(x)

    x = MaxPooling2D((1, 3), name='block3_pool')(x)
    x = Dropout(0.3, name='block3_dropout')(x)

    # Block 4: Abstract features
    x = Conv2D(512, (1, 34), padding='same', name='block4_conv1')(x)
    x = BatchNormalization(name='block4_bn1')(x)
    x = Activation('relu', name='block4_relu1')(x)

    x = Conv2D(512, (1, 32), padding='same', name='block4_conv2')(x)
    x = BatchNormalization(name='block4_bn2')(x)
    x = Activation('relu', name='block4_relu2')(x)

    x = MaxPooling2D((1, 2), name='block4_pool')(x)
    x = Dropout(0.4, name='block4_dropout')(x)

    # Global pooling and classifier
    x = GlobalAveragePooling2D(name='global_avg_pooling')(x)

    # Dense layers with regularization
    x = Dense(1024, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4), name='fc1')(x)
    x = Dropout(0.5, name='fc1_dropout')(x)
    x = BatchNormalization(name='fc1_bn')(x)

    x = Dense(512, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4), name='fc2')(x)
    x = Dropout(0.4, name='fc2_dropout')(x)
    x = BatchNormalization(name='fc2_bn')(x)

    x = Dense(256, activation='relu', name='fc3')(x)
    x = Dropout(0.3, name='fc3_dropout')(x)

    predictions = Dense(9, activation='softmax', name='predictions')(x)

    model = Model(inputs=inputs, outputs=predictions, name='DeepCNN_Character_Recognition')
    return model

def cnn5():
    d = 0.3
    inputs = layers.Input(shape=(2, 4080, 1), name='signal_input')

    # --- Encoder ---
    x1 = layers.Conv2D(10, (1, 51), padding='same')(inputs)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Activation('gelu')(x1)
    x1 = layers.Dropout(d)(x1)

    x2 = layers.Conv2D(20, (1, 51), padding='same')(x1)
    x2 = layers.Activation('gelu')(x2)
    x2 = layers.Dropout(d)(x2)

    x3 = layers.Conv2D(40, (1, 51), padding='same')(x2)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.Activation('gelu')(x3)
    x3 = layers.Dropout(d)(x3)

    x4 = layers.Conv2D(60, (1, 51), padding='same')(x3)
    x4 = layers.Activation('gelu')(x4)
    x4 = layers.Dropout(d)(x4)

    x5 = layers.Conv2D(80, (1, 51), padding='same')(x4)
    x5 = layers.BatchNormalization()(x5)
    x5 = layers.Activation('gelu')(x5)
    x5 = layers.Dropout(d)(x5)

    x6 = layers.Conv2D(80, (1, 51), padding='same')(x5)
    x6 = layers.Activation('gelu')(x6)
    x6 = layers.Dropout(d)(x6)
    # x6 = layers.Concatenate()([x6, x4])   # skip connection 4 ↔ 6

    x7 = layers.Conv2D(100, (1, 40), padding='same')(x6)
    x7 = layers.BatchNormalization()(x7)
    x7 = layers.Activation('gelu')(x7)
    x7 = layers.Dropout(d)(x7)

    x8 = layers.Conv2D(120, (1, 20), padding='same')(x7)
    x8 = layers.Activation('gelu')(x8)
    x8 = layers.Dropout(d)(x8)
    x8 = layers.MaxPooling2D((1, 2), padding='same')(x8)

    # --- Classification Head ---
    x = layers.Flatten()(x8)

    x = layers.Dense(128)(x)
    x = layers.Activation('gelu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(d)(x)

    x = layers.Dense(64)(x)
    x = layers.Activation('gelu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(d)(x)

    output = layers.Dense(9, activation='softmax')(x)

    return Model(inputs=inputs, outputs=output)

def cnn9_skip_connection():
    d = 0.3
    inputs = layers.Input(shape=(2, 4080, 1), name='signal_input')
    x = layers.Conv2D(8, (1, 7), padding='same')(inputs)
    x = layers.Conv2D(16, (1, 7), padding='same')(x)
    x = layers.Conv2D(32, (1, 7), padding='same')(x)

    # --- Encoder ---
    x1 = layers.Conv2D(80, (1, 51), padding='same')(x)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Activation('gelu')(x1)
    x1 = layers.Dropout(d)(x1)
    x1 = layers.MaxPooling2D((1, 2), padding='same')(x1)

    x2 = layers.Conv2D(80, (1, 51), padding='same')(x1)
    x2 = layers.Activation('gelu')(x2)
    x2 = layers.Dropout(d)(x2)
    x2 = layers.MaxPooling2D((1, 2), padding='same')(x2)

    x3 = layers.Conv2D(80, (1, 51), padding='same')(x2)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.Activation('gelu')(x3)
    x3 = layers.Dropout(d)(x3)
    x3 = layers.MaxPooling2D((1, 2), padding='same')(x3)

    x4 = layers.Conv2D(80, (1, 51), padding='same')(x3)
    x4 = layers.Activation('gelu')(x4)
    x4 = layers.Dropout(d)(x4)
    x4 = layers.MaxPooling2D((1, 2), padding='same')(x4)

    x5 = layers.Conv2D(80, (1, 51), padding='same')(x4)
    x5 = layers.BatchNormalization()(x5)
    x5 = layers.Activation('gelu')(x5)
    x5 = layers.Dropout(d)(x5)
    x5 = layers.MaxPooling2D((1, 2), padding='same')(x5)

    x6 = layers.Conv2D(80, (1, 51), padding='same')(x5)
    x6 = layers.Activation('gelu')(x6)
    x6 = layers.Dropout(d)(x6)
    x6 = layers.MaxPooling2D((1, 2), padding='same')(x6)
    # x6 = layers.Concatenate()([x6, x4])   # skip connection 4 ↔ 6

    x7 = layers.Conv2D(80, (1, 51), padding='same')(x6)
    x7 = layers.BatchNormalization()(x7)
    x7 = layers.Activation('gelu')(x7)
    x7 = layers.Dropout(d)(x7)
    x7 = layers.MaxPooling2D((1, 2), padding='same')(x7)
    # x7 = layers.Concatenate()([x7, x3])   # skip connection 3 ↔ 7

    x8 = layers.Conv2D(80, (1, 51), padding='same')(x7)
    x8 = layers.Activation('gelu')(x8)
    x8 = layers.Dropout(d)(x8)
    x8 = layers.MaxPooling2D((1, 2), padding='same')(x8)
    # x8 = layers.Concatenate()([x8, x2])   # skip connection 2 ↔ 8

    x9 = layers.Conv2D(80, (1, 51), padding='same')(x8)
    x9 = layers.BatchNormalization()(x9)
    x9 = layers.Activation('gelu')(x9)
    x9 = layers.Dropout(d)(x9)
    x9 = layers.MaxPooling2D((1, 2), padding='same')(x9)
    # x9 = layers.Concatenate()([x9, x1])   # skip connection 1 ↔ 9

    # --- Classification Head ---
    x = layers.Flatten()(x9)

    x = layers.Dense(128)(x)
    x = layers.Activation('gelu')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(64)(x)
    x = layers.Activation('gelu')(x)
    x = layers.BatchNormalization()(x)

    output = layers.Dense(9, activation='softmax')(x)

    return Model(inputs=inputs, outputs=output)

def inter_encoder_attention(input_length=(2, 4080), num_classes=3):
    inputs = layers.Input(shape=(input_length, 1))

    # Parallel convolutional branches with different kernel sizes
    conv3 = layers.Conv1D(32, 3, padding='same', activation='relu')(inputs)
    conv4 = layers.Conv1D(32, 4, padding='same', activation='relu')(inputs)
    conv6 = layers.Conv1D(32, 6, padding='same', activation='relu')(inputs)
    conv15 = layers.Conv1D(32, 15, padding='same', activation='relu')(inputs)
    conv31 = layers.Conv1D(32, 31, padding='same', activation='relu')(inputs)
    conv63 = layers.Conv1D(32, 63, padding='same', activation='relu')(inputs)
    conv127 = layers.Conv1D(32, 127, padding='same', activation='relu')(inputs)

    # Concatenate all parallel features
    x = layers.concatenate(
        [conv3, conv4, conv6, conv15, conv31, conv63, conv127],
        axis=-1
    )
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool1D(pool_size=2)(x)

    # Attention via 1x1 convolution (Conv1s)
    # Generates channel-wise weights
    attention = layers.Conv1D(
        filters=x.shape[-1], kernel_size=1, padding='same', activation='sigmoid'
    )(x)
    # Apply attention
    x = layers.multiply([x, attention])

    # Further pooling & global feature aggregation
    x = layers.Conv1D(96, 127, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)

    # Fully connected classification head
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs, name='InterEncoder_Attention_CNN')
    return model

def inter_encoder(input_length=12240, num_classes=3):
    inputs = layers.Input(shape=(input_length, 1))

    # Multi-kernel convolution block: kernel sizes 3, 4, 6
    conv3 = layers.Conv1D(64, 3, padding='same', activation='relu')(inputs)
    conv4 = layers.Conv1D(64, 4, padding='same', activation='relu')(inputs)
    conv6 = layers.Conv1D(64, 6, padding='same', activation='relu')(inputs)
    merged = layers.concatenate([conv3, conv4, conv6], axis=-1)
    x = layers.BatchNormalization()(merged)
    x = layers.MaxPool1D(2)(x)

    # Additional hierarchical convolutional layers
    x = layers.Conv1D(128, 15, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool1D(2)(x)

    x = layers.Conv1D(256, 31, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool1D(2)(x)

    x = layers.Conv1D(256, 63, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool1D(2)(x)

    x = layers.Conv1D(96, 127, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)

    # Fully connected layers
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model

def lightweight_comm_net():
    """Efficient model for quick experiments"""
    inputs = layers.Input(shape=(12240, 1), name='signal_input')

    def depthwise_separable_block(x, filters, stride=1):
        # Depthwise convolution
        x = layers.DepthwiseConv1D(7, strides=stride, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('gelu')(x)

        # Pointwise convolution
        x = layers.Conv1D(filters, 1, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('gelu')(x)
        return x

    # Initial standard convolution
    x = layers.Conv1D(32, 15, strides=2, padding='same')(inputs)  # 6120
    x = layers.BatchNormalization()(x)
    x = layers.Activation('gelu')(x)

    # Depthwise separable blocks
    x = depthwise_separable_block(x, 64, stride=2)  # 3060
    x = depthwise_separable_block(x, 128, stride=2)  # 1530
    x = depthwise_separable_block(x, 256, stride=3)  # 510
    x = depthwise_separable_block(x, 512, stride=2)  # 255

    x = layers.Dropout(0.4)(x)

    # Global pooling
    x = layers.GlobalAveragePooling1D()(x)

    # Compact classification head
    x = layers.Dense(128, activation='gelu')(x)
    x = layers.Dropout(0.3)(x)

    output = layers.Dense(9, activation='softmax')(x)

    return Model(inputs=inputs, outputs=output)


def comm_signal_classifier():
    """Optimized for communication signal patterns"""
    inputs = layers.Input(shape=(12240, 1), name='signal_input')

    # Multi-scale bit pattern extraction
    # Short patterns (3-9 bits) - encoder signatures
    short_patterns = layers.Conv1D(64, 3, padding='same', name='short_conv')(inputs)
    short_patterns = layers.Conv1D(64, 5, padding='same')(short_patterns)

    # Medium patterns (15-21 bits) - interleaver signatures
    med_patterns = layers.Conv1D(64, 15, padding='same', name='med_conv')(inputs)
    med_patterns = layers.Conv1D(64, 21, padding='same')(med_patterns)

    # Long patterns (45+ bits) - combined encoder-interleaver effects
    long_patterns = layers.Conv1D(64, 45, padding='same', name='long_conv')(inputs)
    long_patterns = layers.Conv1D(64, 63, padding='same')(long_patterns)

    # Combine multi-scale features
    combined = layers.Concatenate()([short_patterns, med_patterns, long_patterns])
    combined = layers.BatchNormalization()(combined)
    combined = layers.Activation('gelu')(combined)
    combined = layers.Dropout(0.3)(combined)

    # Hierarchical feature learning
    x = layers.Conv1D(128, 9, padding='same')(combined)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('gelu')(x)
    x = layers.MaxPooling1D(2)(x)  # Downsample to 6120

    x = layers.Conv1D(256, 9, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('gelu')(x)
    x = layers.MaxPooling1D(3)(x)  # Downsample to 2040

    x = layers.Conv1D(128, 7, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('gelu')(x)
    x = layers.Dropout(0.4)(x)

    # Global and local feature fusion
    global_feat = layers.GlobalAveragePooling1D()(x)
    max_feat = layers.GlobalMaxPooling1D()(x)
    features = layers.Concatenate()([global_feat, max_feat])

    # Classification head
    x = layers.Dense(512, activation='gelu')(features)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(128, activation='gelu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    output = layers.Dense(9, activation='softmax', name='classification')(x)

    return Model(inputs=inputs, outputs=output)


def modulation_prediction_network():
    """First stage: Predict modulation type (3 classes)"""
    inputs = layers.Input(shape=(4080, 2), name='signal_input')

    p1 = layers.Conv1D(filters=16, kernel_size=3, padding='same')(inputs)
    p2 = layers.Conv1D(filters=16, kernel_size=5, padding='same')(inputs)
    p3 = layers.Conv1D(filters=16, kernel_size=7, padding='same')(inputs)
    p4 = layers.Conv1D(filters=16, kernel_size=1, padding='same')(inputs)

    # Concatenate all paths: 16+16+16+16 = 64 features
    expanded = layers.Concatenate()([p1, p2, p3, p4])

    # Optional: Add activation and normalization
    expanded = layers.BatchNormalization()(expanded)
    expanded = layers.Activation('gelu')(expanded)

    # Initial shared processing
    shared = layers.Conv1D(64, 9, padding='same')(expanded)
    shared = layers.Conv1D(64, 11, padding='same')(shared)
    shared = layers.Conv1D(64, 13, padding='same')(shared)
    shared = layers.Conv1D(64, 15, padding='same')(shared)
    shared = layers.BatchNormalization()(shared)
    shared = layers.Activation(tf.nn.gelu)(shared)
    shared = layers.Dropout(0.2)(shared)

    # Pathway 1 - Innermost (shortest path)
    path1 = layers.Conv1D(32, 15, padding='same')(shared)
    path1 = layers.Conv1D(32, 17, padding='same')(path1)
    path1 = layers.Conv1D(32, 19, padding='same')(path1)
    path1 = layers.Conv1D(32, 21, padding='same')(path1)
    path1 = layers.Activation(tf.nn.gelu)(path1)

    # Pathway 2 - Innermost second (medium-short path)
    path2 = layers.Conv1D(16, 23, padding='same')(path1)
    path2 = layers.Conv1D(16, 25, padding='same')(path2)
    path2 = layers.Conv1D(16, 23, padding='same')(path2)
    path2 = layers.Activation(tf.nn.gelu)(path2)

    # Pathway 3 - Innermost third (medium path) + skip connection from path1
    path3 = layers.Conv1D(32, 21, padding='same')(path2)
    path3 = layers.Conv1D(32, 19, padding='same')(path3)
    path3 = layers.Conv1D(32, 17, padding='same')(path3)
    path3 = layers.Conv1D(32, 15, padding='same')(path3)
    path3 = layers.Activation(tf.nn.gelu)(path3)
    path1_to_path3 = layers.Conv1D(32, 1, padding='same')(path1)
    path3 = layers.Add()([path3, path1_to_path3])

    # Pathway 4 - Outermost (longest path) + skip connection from initial shared layer
    path4 = layers.Conv1D(64, 15, padding='same')(path3)
    path4 = layers.Activation(tf.nn.gelu)(path4)
    path4 = layers.Conv1D(64, 13, padding='same')(path4)
    path4 = layers.Activation(tf.nn.gelu)(path4)
    path4 = layers.Conv1D(64, 11, padding='same', dilation_rate=2)(path4)
    path4 = layers.Activation(tf.nn.gelu)(path4)
    path4 = layers.Conv1D(64, 9, padding='same', dilation_rate=4)(path4)
    path4 = layers.BatchNormalization()(path4)
    path4 = layers.Activation(tf.nn.gelu)(path4)
    shared_to_path4 = layers.Conv1D(64, 1, padding='same')(shared)
    path4 = layers.Add()([path4, shared_to_path4])

    # Global feature extraction
    x = layers.GlobalAveragePooling1D()(path4)

    # Dense layers for modulation prediction
    x = layers.Dense(128, activation='gelu', name='mod_dense_1')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(64, activation='gelu', name='mod_dense_2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    modulation_features = layers.Dense(32, activation='gelu', name='modulation_features')(x)
    modulation_features = layers.BatchNormalization()(modulation_features)

    # Modulation prediction output (3 classes)
    modulation_output = layers.Dense(3, activation='softmax', name='modulation_prediction')(modulation_features)

    return Model(inputs=inputs, outputs=[modulation_output, modulation_features], name='modulation_network')


def encoder_prediction_network():
    """Second stage: Predict encoder type (3 classes) using modulation features"""
    # Input from modulation network features
    modulation_features_input = layers.Input(shape=(32,), name='modulation_features_input')

    # Original signal input for additional processing
    signal_input = layers.Input(shape=(4080, 2), name='signal_input')

    # Light processing of original signal for encoder-specific features
    encoder_conv = layers.Conv1D(16, 15, padding='same')(signal_input)
    encoder_conv = layers.Activation(tf.nn.gelu)(encoder_conv)
    encoder_conv = layers.MaxPooling1D(4, padding='same')(encoder_conv)
    encoder_conv = layers.Conv1D(32, 10, padding='same')(encoder_conv)
    encoder_conv = layers.Activation(tf.nn.gelu)(encoder_conv)
    encoder_conv = layers.GlobalAveragePooling1D()(encoder_conv)

    # Combine modulation features with encoder-specific features
    combined_features = layers.Concatenate()([modulation_features_input, encoder_conv])

    # Dense layers for encoder prediction
    x = layers.Dense(64, activation='gelu')(combined_features)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(32, activation='gelu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    # Encoder prediction output (3 classes)
    encoder_output = layers.Dense(3, activation='softmax', name='encoder_prediction')(x)

    return Model(inputs=[modulation_features_input, signal_input], outputs=encoder_output, name='encoder_network')


def combined_modulation_encoder_model():
    """Combined model that first predicts modulation, then uses those features for encoder prediction"""
    # Input
    signal_input = layers.Input(shape=(4080, 2), name='signal_input')

    # Create modulation network
    mod_network = modulation_prediction_network()

    # Get modulation prediction and features
    modulation_pred, modulation_features = mod_network(signal_input)

    # Create encoder network
    enc_network = encoder_prediction_network()

    # Get encoder prediction using modulation features
    encoder_pred = enc_network([modulation_features, signal_input])

    # Combined model
    return Model(inputs=signal_input,
                 outputs=[modulation_pred, encoder_pred],
                 name='combined_mod_enc_model')


def feature_net_6():
    """Simplified smaller model with reduced parameters"""
    inputs = layers.Input(shape=(12240, 1), name='signal_input')

    # STAGE 1: SMALL - Reduced multi-scale feature expansion (1 -> 48)
    p1 = layers.Conv1D(filters=16, kernel_size=3, padding='same')(inputs)
    p2 = layers.Conv1D(filters=16, kernel_size=5, padding='same')(inputs)
    p3 = layers.Conv1D(filters=16, kernel_size=1, padding='same')(inputs)

    expanded = layers.Concatenate()([p1, p2, p3])  # 48 features (reduced from 128)
    expanded = layers.BatchNormalization()(expanded)
    expanded = layers.Activation('gelu')(expanded)

    # STAGE 2: BIG - Reduced shared processing (48 -> 64)
    shared = layers.Conv1D(32, 7, padding='same')(expanded)  # Reduced from 128
    shared = layers.Conv1D(32, 9, padding='same')(shared)  # Reduced kernel size
    shared = layers.BatchNormalization()(shared)
    shared = layers.Activation(tf.nn.gelu)(shared)
    shared = layers.Dropout(0.25)(shared)  # Reduced dropout

    # STAGE 3: SMALL - Pathway 1 (64 -> 32)
    path1 = layers.Conv1D(16, 7, padding='same')(shared)  # Reduced from 64
    path1 = layers.Conv1D(16, 7, padding='same')(path1)  # Reduced kernel size
    path1 = layers.Activation(tf.nn.gelu)(path1)
    path1 = layers.BatchNormalization()(path1)

    # STAGE 4: SMALLER - Pathway 2 (32 -> 16)
    path2 = layers.Conv1D(8, 7, padding='same')(path1)  # Reduced from 32
    path2 = layers.Conv1D(8, 7, padding='same')(path2)
    path2 = layers.Activation(tf.nn.gelu)(path2)
    path2 = layers.BatchNormalization()(path2)

    # STAGE 5: BIG AGAIN - Pathway 3 (16 -> 32) + skip
    path3 = layers.Conv1D(16, 7, padding='same')(path2)  # Reduced from 64
    path3 = layers.Conv1D(16, 7, padding='same')(path3)
    path3 = layers.Activation(tf.nn.gelu)(path3)
    path3 = layers.BatchNormalization()(path3)

    # Skip connection: path1 to path3
    path1_to_path3 = layers.Conv1D(16, 1, padding='same')(path1)
    path3 = layers.Add()([path3, path1_to_path3])

    # STAGE 6: BIGGEST - Pathway 4 (32 -> 64) + skip
    path4 = layers.Conv1D(32, 7, padding='same')(path3)  # Reduced from 128
    path4 = layers.Conv1D(32, 7, padding='same')(path4)
    path4 = layers.BatchNormalization()(path4)
    path4 = layers.Activation(tf.nn.gelu)(path4)

    # Skip connection: shared to path4
    shared_to_path4 = layers.Conv1D(32, 1, padding='same')(shared)
    path4 = layers.Add()([path4, shared_to_path4])
    path4 = layers.BatchNormalization()(path4)
    path4 = layers.Activation(tf.nn.gelu)(path4)
    path4 = layers.Dropout(0.3)(path4)  # Reduced dropout

    # Global feature extraction
    x = layers.GlobalAveragePooling1D()(path4)

    # Reduced dense layers
    x = layers.Dense(128, activation='gelu', name='mod_dense_1')(x)  # Reduced from 256
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(64, activation='gelu', name='mod_dense_2')(x)  # Reduced from 128
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    # Removed one dense layer to make it smaller

    # Output (9 classes)
    output = layers.Dense(9, activation='softmax', name='modulation_prediction')(x)

    return Model(inputs=inputs, outputs=output)

def feature_net_5():
    """Enhanced model with small-to-big-to-small-to-big approach"""
    inputs = layers.Input(shape=(12240, 1), name='signal_input')

    # STAGE 1: SMALL - Multi-scale feature expansion (2 -> 128)
    p1 = layers.Conv1D(filters=32, kernel_size=3, padding='same')(inputs)
    p2 = layers.Conv1D(filters=32, kernel_size=5, padding='same')(inputs)
    p3 = layers.Conv1D(filters=32, kernel_size=7, padding='same')(inputs)
    p4 = layers.Conv1D(filters=32, kernel_size=1, padding='same')(inputs)

    expanded = layers.Concatenate()([p1, p2, p3, p4])  # 128 features
    expanded = layers.BatchNormalization()(expanded)
    expanded = layers.Activation('gelu')(expanded)

    # STAGE 2: BIG - Initial shared processing (128 -> 256)
    shared = layers.Conv1D(128, 9, padding='same')(expanded)
    shared = layers.Conv1D(128, 11, padding='same')(shared)
    shared = layers.BatchNormalization()(shared)
    shared = layers.Activation(tf.nn.gelu)(shared)
    shared = layers.Dropout(0.3)(shared)

    # STAGE 3: SMALL - Pathway 1 (256 -> 128)
    path1 = layers.Conv1D(64, 11, padding='same')(shared)
    path1 = layers.Conv1D(64, 11, padding='same')(path1)
    path1 = layers.Activation(tf.nn.gelu)(path1)
    path1 = layers.BatchNormalization()(path1)

    # STAGE 4: SMALLER - Pathway 2 (128 -> 64)
    path2 = layers.Conv1D(32, 11, padding='same')(path1)
    path2 = layers.Conv1D(32, 11, padding='same')(path2)
    path2 = layers.Activation(tf.nn.gelu)(path2)
    path2 = layers.BatchNormalization()(path2)

    # STAGE 5: BIG AGAIN - Pathway 3 (64 -> 128) + skip from path1
    path3 = layers.Conv1D(64, 11, padding='same')(path2)
    path3 = layers.Conv1D(64, 11, padding='same')(path3)
    path3 = layers.Activation(tf.nn.gelu)(path3)
    path3 = layers.BatchNormalization()(path3)

    # Skip connection: path1 to path3
    path1_to_path3 = layers.Conv1D(64, 1, padding='same')(path1)
    path3 = layers.Add()([path3, path1_to_path3])

    # STAGE 6: BIGGEST - Pathway 4 (128 -> 256) + skip from shared
    path4 = layers.Conv1D(128, 11, padding='same')(path3)
    path4 = layers.Conv1D(128, 11, padding='same')(path4)
    path4 = layers.BatchNormalization()(path4)
    path4 = layers.Activation(tf.nn.gelu)(path4)

    # Skip connection: shared to path4
    shared_to_path4 = layers.Conv1D(128, 1, padding='same')(shared)
    path4 = layers.Add()([path4, shared_to_path4])
    path4 = layers.BatchNormalization()(path4)
    path4 = layers.Activation(tf.nn.gelu)(path4)
    path4 = layers.Dropout(0.4)(path4)

    # Global feature extraction
    x = layers.GlobalAveragePooling1D()(path4)

    x = layers.Dense(256, activation='gelu', name='mod_dense_1')(x)  # BIG
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(128, activation='gelu', name='mod_dense_2')(x)  # SMALLER
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(64, activation='gelu', name='mod_dense_3')(x)  # BIG AGAIN
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # Output (9 classes)
    output = layers.Dense(9, activation='softmax', name='modulation_prediction')(x)

    return Model(inputs=inputs, outputs=output)

def improved_feature_net_3():
    inputs = layers.Input(shape=(4080, 2, 1), name='signal_input')

    # Convert to 1D - more suitable for time series
    x = layers.Reshape((4080, 2))(inputs)

    # Progressive feature extraction (like your feature_net_2)
    x = layers.Conv1D(32, 15, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(64, 9, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(128, 5, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    # Add dilated convolutions for better pattern recognition
    x1 = layers.Conv1D(64, 10, padding='same', dilation_rate=1)(x)
    x2 = layers.Conv1D(64, 15, padding='same', dilation_rate=2)(x)
    x3 = layers.Conv1D(64, 20, padding='same', dilation_rate=4)(x)

    combined = layers.Concatenate()([x1, x2, x3])
    combined = layers.Dropout(0.3)(combined)

    # Global pooling
    x = layers.GlobalAveragePooling1D()(combined)

    # Classifier
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(9, activation='softmax')(x)

    return Model(inputs=inputs, outputs=outputs)

def rnn():
    inputs = layers.Input(shape=(12240, 1), name='signal_input')
    r = layers.LSTM(64, return_sequences=True, name='lstm1')(inputs)
    r = layers.LSTM(64, name='lstm2')(r)
    x_final = layers.Dense(units=128, activation='relu', name='dense_128')(r)
    x_final = layers.BatchNormalization(axis=1, momentum=0.99, name='bn_dense_128')(x_final)
    x_final = layers.Dropout(rate=0.3, name='dropout_dense_128')(x_final)

    x_final = layers.Dense(units=48, activation='relu', name='dense_48')(x_final)
    x_final = layers.BatchNormalization(axis=1, momentum=0.99, name='bn_dense_48')(x_final)
    x_final = layers.Dropout(rate=0.3, name='dropout_dense_48')(x_final)

    # Output layer
    outputs = layers.Dense(units=9, activation='softmax', name='output_softmax')(x_final)
    model = Model(inputs=inputs, outputs=outputs)

    return model

def feature_net_3():
    """
    Conv1D version of feature_net_3 with single attention mechanism
    Input shape: (None, 4080, 2)
    Output: 9-class classification with softmax
    """
    # Input layer
    signal_input = layers.Input(shape=(12240, 1), name='signal_input')

    # Initial Conv1D layers (4 consecutive convolutions)
    x = layers.Conv1D(filters=64, kernel_size=7, strides=1, padding='same', activation='gelu', name='conv1d_initial')(
        signal_input)
    x = layers.Conv1D(filters=64, kernel_size=7, strides=1, padding='same', activation='gelu', name='conv1d_1')(x)
    x = layers.Conv1D(filters=64, kernel_size=7, strides=1, padding='same', activation='gelu', name='conv1d_2')(x)
    x = layers.Conv1D(filters=64, kernel_size=7, strides=1, padding='same', activation='gelu', name='conv1d_3')(x)
    x = layers.LeakyReLU(alpha=0.3, name='leaky_relu_branch')(x)
    x = layers.MaxPooling1D(3)(x)

    # ===== ALL FEATURE EXTRACTION LAYERS (NO GROUP ATTENTION) ===== # Include base features

    # x25
    x25 = layers.Conv1D(filters=102, kernel_size=10, strides=1, padding='same', activation='gelu', name='conv1d_x25')(x)
    x25 = layers.Conv1D(filters=102, kernel_size=10, strides=1, padding='same', activation='gelu', name='conv1d_x26')(x25)
    x25 = layers.Conv1D(filters=102, kernel_size=10, strides=1, padding='same', activation='gelu', name='conv1d_x27')(x25)
    x25 = layers.Conv1D(filters=102, kernel_size=10, strides=1, padding='same', activation='gelu', name='conv1d_x28')(x25)
    x25 = layers.BatchNormalization(axis=2, momentum=0.99, name='bn_x25')(x25)
    x25 = layers.LeakyReLU(alpha=0.3, name='leaky_relu_x25')(x25)

    # x1
    x1 = layers.Conv1D(filters=64, kernel_size=16, strides=1, padding='same', activation='gelu', name='conv1d_x1')(x)
    x1 = layers.Conv1D(filters=64, kernel_size=16, strides=1, padding='same', activation='gelu', name='conv1d_x2')(x1)
    x1 = layers.Conv1D(filters=64, kernel_size=16, strides=1, padding='same', activation='gelu', name='conv1d_x3')(x1)
    x1 = layers.Conv1D(filters=64, kernel_size=16, strides=1, padding='same', activation='gelu', name='conv1d_x4')(x1)
    x1 = layers.BatchNormalization(axis=2, momentum=0.99, name='bn_x1')(x1)
    x1 = layers.LeakyReLU(alpha=0.3, name='leaky_relu_x1')(x1)

    # x5
    x5 = layers.Conv1D(filters=45, kernel_size=24, strides=1, padding='same', activation='gelu', name='conv1d_x5')(x)
    x5 = layers.Conv1D(filters=45, kernel_size=24, strides=1, padding='same', activation='gelu', name='conv1d_x6')(x5)
    x5 = layers.Conv1D(filters=45, kernel_size=24, strides=1, padding='same', activation='gelu', name='conv1d_x7')(x5)
    x5 = layers.Conv1D(filters=45, kernel_size=24, strides=1, padding='same', activation='gelu', name='conv1d_x8')(x5)
    x5 = layers.BatchNormalization(axis=2, momentum=0.99, name='bn_x5')(x5)
    x5 = layers.LeakyReLU(alpha=0.3, name='leaky_relu_x5')(x5)

    # x9
    x9 = layers.Conv1D(filters=26, kernel_size=40, strides=1, padding='same', activation='gelu', name='conv1d_x9')(x)
    x9 = layers.Conv1D(filters=26, kernel_size=40, strides=1, padding='same', activation='gelu', name='conv1d_x10')(x9)
    x9 = layers.Conv1D(filters=26, kernel_size=40, strides=1, padding='same', activation='gelu', name='conv1d_x11')(x9)
    x9 = layers.Conv1D(filters=26, kernel_size=40, strides=1, padding='same', activation='gelu', name='conv1d_x12')(x9)
    x9 = layers.BatchNormalization(axis=2, momentum=0.99, name='bn_x9')(x9)
    x9 = layers.LeakyReLU(alpha=0.3, name='leaky_relu_x9')(x9)

    # x13
    x13 = layers.Conv1D(filters=20, kernel_size=50, strides=1, padding='same', activation='gelu', name='conv1d_x13')(x)
    x13 = layers.Conv1D(filters=20, kernel_size=50, strides=1, padding='same', activation='gelu', name='conv1d_x14')(x13)
    x13 = layers.Conv1D(filters=20, kernel_size=50, strides=1, padding='same', activation='gelu', name='conv1d_x15')(x13)
    x13 = layers.Conv1D(filters=20, kernel_size=50, strides=1, padding='same', activation='gelu', name='conv1d_x16')(x13)
    x13 = layers.BatchNormalization(axis=2, momentum=0.99, name='bn_x13')(x13)
    x13 = layers.LeakyReLU(alpha=0.3, name='leaky_relu_x13')(x13)

    # x15
    x15 = layers.Conv1D(filters=16, kernel_size=64, strides=1, padding='same', activation='gelu', name='conv1d_x17')(x)
    x15 = layers.Conv1D(filters=16, kernel_size=64, strides=1, padding='same', activation='gelu', name='conv1d_x18')(x15)
    x15 = layers.Conv1D(filters=16, kernel_size=64, strides=1, padding='same', activation='gelu', name='conv1d_x19')(x15)
    x15 = layers.Conv1D(filters=16, kernel_size=64, strides=1, padding='same', activation='gelu', name='conv1d_x20')(x15)
    x15 = layers.BatchNormalization(axis=2, momentum=0.99, name='bn_x15')(x15)
    x15 = layers.LeakyReLU(alpha=0.3, name='leaky_relu_x15')(x15)


    # x18 (specified: filters=48, kernel_size=85)
    x18 = layers.Conv1D(filters=12, kernel_size=85, strides=1, padding='same', activation='gelu', name='conv1d_x21')(x)
    x18 = layers.Conv1D(filters=12, kernel_size=85, strides=1, padding='same', activation='gelu', name='conv1d_x22')(x18)
    x18 = layers.Conv1D(filters=12, kernel_size=85, strides=1, padding='same', activation='gelu', name='conv1d_x23')(x18)
    x18 = layers.Conv1D(filters=12, kernel_size=85, strides=1, padding='same', activation='gelu', name='conv1d_x24')(x18)
    x18 = layers.BatchNormalization(axis=2, momentum=0.99, name='bn_x18')(x18)
    x18 = layers.LeakyReLU(alpha=0.3, name='leaky_relu_x18')(x18)

    # ===== SINGLE ATTENTION MECHANISM ON ALL FEATURES =====
    # Concatenate all features
    all_features_concat = layers.Concatenate(name='all_features_concat')([x,x25,x1,x5,x9,x13,x15,x18])
    all_features_dropout = layers.Dropout(0.2, name='all_features_dropout')(all_features_concat)

    # Calculate total filters
    total_filters = 64 + 102 + 64 + 45 + 26 + 20 + 16 + 12

    # Single attention mechanism
    attention = layers.Conv1D(total_filters, 1, activation='sigmoid', name='attention')(all_features_dropout)
    attended_features = layers.Multiply(name='attention_multiply')([all_features_dropout, attention])

    # Residual connection
    final_features = layers.Add(name='residual_add')([all_features_dropout, attended_features])
    final_features = layers.Dropout(0.2, name='final_dropout')(final_features)

    # Global Average Pooling
    x_final = layers.GlobalAveragePooling1D(name='global_avg_pool')(final_features)

    # Dense layers
    x_final = layers.Dense(units=256, activation='relu', name='dense_256')(x_final)
    x_final = layers.BatchNormalization(axis=1, momentum=0.99, name='bn_dense_256')(x_final)
    x_final = layers.Dropout(rate=0.3, name='dropout_dense_256')(x_final)

    x_final = layers.Dense(units=128, activation='relu', name='dense_128')(x_final)
    x_final = layers.BatchNormalization(axis=1, momentum=0.99, name='bn_dense_128')(x_final)
    x_final = layers.Dropout(rate=0.3, name='dropout_dense_128')(x_final)

    x_final = layers.Dense(units=96, activation='relu', name='dense_96')(x_final)
    x_final = layers.BatchNormalization(axis=1, momentum=0.99, name='bn_dense_96')(x_final)
    x_final = layers.Dropout(rate=0.3, name='dropout_dense_96')(x_final)

    x_final = layers.Dense(units=48, activation='relu', name='dense_48')(x_final)
    x_final = layers.BatchNormalization(axis=1, momentum=0.99, name='bn_dense_48')(x_final)
    x_final = layers.Dropout(rate=0.3, name='dropout_dense_48')(x_final)

    # Output layer
    outputs = layers.Dense(units=9, activation='softmax', name='output_softmax')(x_final)

    # Create model
    model = Model(inputs=signal_input, outputs=outputs, name='feature_net_3_single_attention')

    return model

def residual_conv_block(x, filters, kernel_size, name_prefix):
    shortcut = x
    x = layers.Conv1D(filters, kernel_size, padding="same", activation=None, name=f"{name_prefix}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
    x = layers.Activation("gelu", name=f"{name_prefix}_act1")(x)
    x = layers.Conv1D(filters, kernel_size, padding="same", activation=None, name=f"{name_prefix}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)
    x = layers.Add(name=f"{name_prefix}_add")([shortcut, x]) if shortcut.shape[-1] == filters else x
    x = layers.Activation("gelu", name=f"{name_prefix}_out")(x)
    return x

def feature_net_3_improved():
    signal_input = layers.Input(shape=(4080, 2), name="signal_input")

    # Initial feature extraction
    x = layers.Conv1D(64, 7, padding="same", activation="gelu")(signal_input)
    x = layers.MaxPooling1D(3)(x)

    # Multi-scale residual branches
    branches = []
    for i, (filters, k) in enumerate([(102,10),(64,16),(45,24),(26,40),(20,50),(16,64),(12,85)]):
        b = residual_conv_block(x, filters, k, f"branch{i}")
        branches.append(b)

    # Concatenate
    all_features = layers.Concatenate()(branches + [x])

    # Multi-head attention instead of single gate
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=32)(all_features, all_features)
    x = layers.Add()([all_features, attn])
    x = layers.LayerNormalization()(x)

    # Attention pooling
    query = layers.Dense(x.shape[-1], activation="tanh")(x)
    weights = layers.Softmax(axis=1)(query)
    x = tf.reduce_sum(x * weights, axis=1)

    # Classifier head
    x_final = layers.Dense(256, activation="swish", name="dense_256")(x)
    x_final = layers.BatchNormalization(name="bn_256")(x_final)
    x_final = layers.Dropout(0.3, name="dropout_256")(x_final)

    # Residual block style
    shortcut = x_final
    x_res = layers.Dense(128, activation="swish", name="dense_128")(x_final)
    x_res = layers.BatchNormalization(name="bn_128")(x_res)
    x_res = layers.Dropout(0.3, name="dropout_128")(x_res)
    x_final = layers.Add(name="residual_128")([shortcut, x_res]) if shortcut.shape[-1] == 128 else x_res

    # Pyramid compression
    x_final = layers.Dense(64, activation="swish", name="dense_64")(x_final)
    x_final = layers.Dropout(0.3, name="dropout_64")(x_final)
    x_final = layers.Dense(32, activation="swish", name="dense_32")(x_final)

    # Final norm before softmax
    x_final = layers.LayerNormalization(name="final_norm")(x_final)

    # Output
    outputs = layers.Dense(9, activation="softmax", name="output_softmax")(x_final)

    return Model(signal_input, outputs, name="feature_net_3_improved")


def feature_net_4():
    """
    ConvLSTM1D version of feature_net_3 with single attention mechanism
    Input shape: (None, 4080, 2)
    Output: 9-class classification with softmax
    """
    # Input layer
    signal_input = layers.Input(shape=(4080, 2), name='signal_input')
    x = layers.Reshape((4080, 1, 2), name='reshape_4d')(signal_input)

    # Initial ConvLSTM1D layers (4 consecutive convolutions)
    x = layers.ConvLSTM1D(filters=64, kernel_size=7, strides=1, padding='same',
                          activation='gelu', return_sequences=True, name='convlstm1d_initial')(x)
    x = layers.ConvLSTM1D(filters=64, kernel_size=7, strides=1, padding='same',
                          activation='gelu', return_sequences=True, name='convlstm1d_1')(x)
    x = layers.ConvLSTM1D(filters=64, kernel_size=7, strides=1, padding='same',
                          activation='gelu', return_sequences=True, name='convlstm1d_2')(x)
    x = layers.ConvLSTM1D(filters=64, kernel_size=7, strides=1, padding='same',
                          activation='gelu', return_sequences=True, name='convlstm1d_3')(x)
    x = layers.LeakyReLU(alpha=0.3, name='leaky_relu_branch')(x)
    x = layers.MaxPooling2D((3, 1))(x)

    # ===== ALL FEATURE EXTRACTION LAYERS WITH CONVLSTM1D =====

    # x25
    # x25 = layers.ConvLSTM1D(filters=102, kernel_size=10, strides=1, padding='same',
    #                         activation='gelu', return_sequences=True, name='convlstm1d_x25')(x)
    # x25 = layers.ConvLSTM1D(filters=102, kernel_size=10, strides=1, padding='same',
    #                         activation='gelu', return_sequences=True, name='convlstm1d_x26')(x25)
    # x25 = layers.ConvLSTM1D(filters=102, kernel_size=10, strides=1, padding='same',
    #                         activation='gelu', return_sequences=True, name='convlstm1d_x27')(x25)
    # x25 = layers.ConvLSTM1D(filters=102, kernel_size=10, strides=1, padding='same',
    #                         activation='gelu', return_sequences=True, name='convlstm1d_x28')(x25)
    # x25 = layers.BatchNormalization(axis=2, momentum=0.99, name='bn_x25')(x25)
    # x25 = layers.LeakyReLU(alpha=0.3, name='leaky_relu_x25')(x25)

    # x1
    # x1 = layers.ConvLSTM1D(filters=64, kernel_size=16, strides=1, padding='same',
    #                        activation='gelu', return_sequences=True, name='convlstm1d_x1')(x)
    # x1 = layers.ConvLSTM1D(filters=64, kernel_size=16, strides=1, padding='same',
    #                        activation='gelu', return_sequences=True, name='convlstm1d_x2')(x1)
    # x1 = layers.ConvLSTM1D(filters=64, kernel_size=16, strides=1, padding='same',
    #                        activation='gelu', return_sequences=True, name='convlstm1d_x3')(x1)
    # x1 = layers.ConvLSTM1D(filters=64, kernel_size=16, strides=1, padding='same',
    #                        activation='gelu', return_sequences=True, name='convlstm1d_x4')(x1)
    # x1 = layers.BatchNormalization(axis=2, momentum=0.99, name='bn_x1')(x1)
    # x1 = layers.LeakyReLU(alpha=0.3, name='leaky_relu_x1')(x1)
    #
    # # x5
    # x5 = layers.ConvLSTM1D(filters=45, kernel_size=24, strides=1, padding='same',
    #                        activation='gelu', return_sequences=True, name='convlstm1d_x5')(x)
    # x5 = layers.ConvLSTM1D(filters=45, kernel_size=24, strides=1, padding='same',
    #                        activation='gelu', return_sequences=True, name='convlstm1d_x6')(x5)
    # x5 = layers.ConvLSTM1D(filters=45, kernel_size=24, strides=1, padding='same',
    #                        activation='gelu', return_sequences=True, name='convlstm1d_x7')(x5)
    # x5 = layers.ConvLSTM1D(filters=45, kernel_size=24, strides=1, padding='same',
    #                        activation='gelu', return_sequences=True, name='convlstm1d_x8')(x5)
    # x5 = layers.BatchNormalization(axis=2, momentum=0.99, name='bn_x5')(x5)
    # x5 = layers.LeakyReLU(alpha=0.3, name='leaky_relu_x5')(x5)

    # x9
    x9 = layers.ConvLSTM1D(filters=26, kernel_size=40, strides=1, padding='same',
                           activation='gelu', return_sequences=True, name='convlstm1d_x9')(x)
    x9 = layers.ConvLSTM1D(filters=26, kernel_size=40, strides=1, padding='same',
                           activation='gelu', return_sequences=True, name='convlstm1d_x10')(x9)
    x9 = layers.ConvLSTM1D(filters=26, kernel_size=40, strides=1, padding='same',
                           activation='gelu', return_sequences=True, name='convlstm1d_x11')(x9)
    x9 = layers.ConvLSTM1D(filters=26, kernel_size=40, strides=1, padding='same',
                           activation='gelu', return_sequences=True, name='convlstm1d_x12')(x9)
    x9 = layers.BatchNormalization(axis=2, momentum=0.99, name='bn_x9')(x9)
    x9 = layers.LeakyReLU(alpha=0.3, name='leaky_relu_x9')(x9)

    # x13
    x13 = layers.ConvLSTM1D(filters=20, kernel_size=50, strides=1, padding='same',
                            activation='gelu', return_sequences=True, name='convlstm1d_x13')(x)
    x13 = layers.ConvLSTM1D(filters=20, kernel_size=50, strides=1, padding='same',
                            activation='gelu', return_sequences=True, name='convlstm1d_x14')(x13)
    x13 = layers.ConvLSTM1D(filters=20, kernel_size=50, strides=1, padding='same',
                            activation='gelu', return_sequences=True, name='convlstm1d_x15')(x13)
    x13 = layers.ConvLSTM1D(filters=20, kernel_size=50, strides=1, padding='same',
                            activation='gelu', return_sequences=True, name='convlstm1d_x16')(x13)
    x13 = layers.BatchNormalization(axis=2, momentum=0.99, name='bn_x13')(x13)
    x13 = layers.LeakyReLU(alpha=0.3, name='leaky_relu_x13')(x13)

    # x15
    x15 = layers.ConvLSTM1D(filters=16, kernel_size=64, strides=1, padding='same',
                            activation='gelu', return_sequences=True, name='convlstm1d_x17')(x)
    x15 = layers.ConvLSTM1D(filters=16, kernel_size=64, strides=1, padding='same',
                            activation='gelu', return_sequences=True, name='convlstm1d_x18')(x15)
    x15 = layers.ConvLSTM1D(filters=16, kernel_size=64, strides=1, padding='same',
                            activation='gelu', return_sequences=True, name='convlstm1d_x19')(x15)
    x15 = layers.ConvLSTM1D(filters=16, kernel_size=64, strides=1, padding='same',
                            activation='gelu', return_sequences=True, name='convlstm1d_x20')(x15)
    x15 = layers.BatchNormalization(axis=2, momentum=0.99, name='bn_x15')(x15)
    x15 = layers.LeakyReLU(alpha=0.3, name='leaky_relu_x15')(x15)

    # x18
    # x18 = layers.ConvLSTM1D(filters=12, kernel_size=85, strides=1, padding='same',
    #                         activation='gelu', return_sequences=True, name='convlstm1d_x21')(x)
    # x18 = layers.ConvLSTM1D(filters=12, kernel_size=85, strides=1, padding='same',
    #                         activation='gelu', return_sequences=True, name='convlstm1d_x22')(x18)
    # x18 = layers.ConvLSTM1D(filters=12, kernel_size=85, strides=1, padding='same',
    #                         activation='gelu', return_sequences=True, name='convlstm1d_x23')(x18)
    # x18 = layers.ConvLSTM1D(filters=12, kernel_size=85, strides=1, padding='same',
    #                         activation='gelu', return_sequences=True, name='convlstm1d_x24')(x18)
    # x18 = layers.BatchNormalization(axis=2, momentum=0.99, name='bn_x18')(x18)
    # x18 = layers.LeakyReLU(alpha=0.3, name='leaky_relu_x18')(x18)

    # ===== SINGLE ATTENTION MECHANISM ON ALL FEATURES =====
    # Concatenate all features
    all_features_concat = layers.Concatenate(name='all_features_concat')([x,x9,x13,x15])
    all_features_dropout = layers.Dropout(0.2, name='all_features_dropout')(all_features_concat)

    # Calculate total filters
    total_filters = 64 + 26 + 20 + 16

    # Single attention mechanism (using Conv1D for attention weights)
    attention = layers.Conv2D(total_filters, (1, 1), activation='sigmoid', name='attention')(all_features_dropout)
    attended_features = layers.Multiply(name='attention_multiply')([all_features_dropout, attention])

    # Residual connection
    final_features = layers.Add(name='residual_add')([all_features_dropout, attended_features])
    final_features = layers.Dropout(0.2, name='final_dropout')(final_features)

    # Global Average Pooling
    x_final = layers.GlobalAveragePooling2D(name='global_avg_pool')(final_features)

    # Continue with your RNN branch and final layers...
    # RNN model (keeping original structure or using ConvLSTM here too)
    # r = layers.LSTM(64, return_sequences=True, name='lstm1')(x)
    # r = layers.LSTM(64, name='lstm2')(r)
    # r = layers.Dense(64, activation='gelu')(r)
    #
    # x_final = layers.Concatenate(name='final_concat')([x_final, r])
    # x_final = layers.Dropout(0.2, name='x_final_dropout')(x_final)

    # Dense layers
    x_final = layers.Dense(units=128, activation='relu', name='dense_128')(x_final)
    x_final = layers.BatchNormalization(axis=1, momentum=0.99, name='bn_dense_128')(x_final)
    x_final = layers.Dropout(rate=0.3, name='dropout_dense_128')(x_final)

    x_final = layers.Dense(units=48, activation='relu', name='dense_48')(x_final)
    x_final = layers.BatchNormalization(axis=1, momentum=0.99, name='bn_dense_48')(x_final)
    x_final = layers.Dropout(rate=0.3, name='dropout_dense_48')(x_final)

    # Output layer
    outputs = layers.Dense(units=9, activation='softmax', name='output_softmax')(x_final)

    # Create model
    model = Model(inputs=signal_input, outputs=outputs, name='feature_net_3_convlstm')

    return model

def build_improved_1d_cnn(input_shape=(4080, 2), num_classes=3, dropout_rate=0.4):
    inputs = layers.Input(shape=input_shape)

    # Initial convolution
    x = layers.Conv1D(64, 7, padding='same', strides=2)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(3, strides=2, padding='same')(x)

    # Improved separable block with proper residual handling
    def sep_block(x, filters, kernel_size, strides=1):
        residual = x

        x = layers.SeparableConv1D(filters, kernel_size, padding='same', strides=strides)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv1D(filters, kernel_size, padding='same')(x)
        x = layers.BatchNormalization()(x)

        # Handle both stride and filter dimension changes
        if strides > 1 or residual.shape[-1] != filters:
            residual = layers.Conv1D(filters, 1, strides=strides, padding='same')(residual)
            residual = layers.BatchNormalization()(residual)

        x = layers.Add()([x, residual])
        x = layers.Activation('relu')(x)
        return x

    # Shared feature extraction
    x = sep_block(x, 128, 5, strides=2)
    x = sep_block(x, 128, 5)
    x = sep_block(x, 256, 3, strides=2)
    x = sep_block(x, 256, 3)

    # Task-specific feature extraction
    # Encoder branch
    x_encoder = sep_block(x, 512, 3, strides=2)
    x_encoder = sep_block(x_encoder, 512, 3)
    x_encoder = layers.GlobalAveragePooling1D()(x_encoder)
    x_encoder = layers.Dropout(dropout_rate)(x_encoder)
    x_encoder = layers.Dense(256, activation='relu')(x_encoder)
    x_encoder = layers.Dropout(dropout_rate)(x_encoder)
    outputs1 = layers.Dense(num_classes, activation='softmax', name='encoder_out')(x_encoder)

    # Interleaver branch - different architecture
    x_interleaver = sep_block(x, 384, 5, strides=2)  # Different filter size
    x_interleaver = sep_block(x_interleaver, 384, 5)
    x_interleaver = layers.GlobalAveragePooling1D()(x_interleaver)
    x_interleaver = layers.Dropout(dropout_rate)(x_interleaver)
    x_interleaver = layers.Dense(192, activation='relu')(x_interleaver)  # Different dense size
    x_interleaver = layers.Dropout(dropout_rate)(x_interleaver)
    outputs2 = layers.Dense(num_classes, activation='softmax', name='interleaver_out')(x_interleaver)

    return Model(inputs, {'encoder_out': outputs1, 'interleaver_out': outputs2})

def enhanced_feature_net_2():
    """
    Your feature_net_2 enhanced with additional capabilities
    """
    inputs = layers.Input(shape=(4080, 2), name='signal_input')

    # === Your Original Architecture (Keep This!) ===
    x = layers.Conv1D(32, 9, padding='same')(inputs)
    x = layers.Conv1D(32, 9, padding='same')(x)
    x = layers.Conv1D(32, 9, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(tf.nn.gelu)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.MaxPooling1D(2, padding='same')(x)

    x = layers.Conv1D(40, 10, padding='same')(x)
    x = layers.Activation(tf.nn.gelu)(x)

    x = layers.Conv1D(48, 15, padding='same')(x)
    x = layers.Activation(tf.nn.gelu)(x)

    x = layers.Conv1D(48, 20, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2, padding='same')(x)
    x = layers.Activation(tf.nn.gelu)(x)

    # Enhanced dilated convolutions with more scales
    x1 = layers.Conv1D(48, 20, padding='same', dilation_rate=1)(x)
    x2 = layers.Conv1D(48, 25, padding='same', dilation_rate=2)(x)
    x3 = layers.Conv1D(48, 30, padding='same', dilation_rate=4)(x)
    x4 = layers.Conv1D(48, 35, padding='same', dilation_rate=8)(x)  # Added scale

    dilated_combined = layers.Concatenate()([x, x1, x2, x3, x4])
    dilated_combined = layers.Dropout(0.2)(dilated_combined)

    # Enhanced attention with multi-head concept
    attention = layers.Conv1D(240, 1, activation='sigmoid')(dilated_combined)  # 240 = 48*5
    attended_features = layers.Multiply()([dilated_combined, attention])

    # Add residual connection for better gradient flow
    attended_features = layers.Add()([dilated_combined, attended_features])
    attended_features = layers.Dropout(0.2)(attended_features)

    # === New: Dual-path processing ===
    # Path 1: Your original global pooling
    global_features = layers.GlobalAveragePooling1D()(attended_features)

    # Path 2: Local feature extraction for fine details
    local_conv = layers.Conv1D(64, 1)(attended_features)
    local_features = layers.GlobalMaxPooling1D()(local_conv)

    # Combine both paths
    combined_features = layers.Concatenate()([global_features, local_features])

    # Enhanced dense layers with skip connections
    dense1 = layers.Dense(128, activation='gelu')(combined_features)
    dense1 = layers.BatchNormalization()(dense1)
    dense1 = layers.Dropout(0.3)(dense1)

    dense2 = layers.Dense(64, activation='gelu')(dense1)
    dense2 = layers.BatchNormalization()(dense2)
    dense2 = layers.Dropout(0.3)(dense2)

    # Skip connection
    dense2_skip = layers.Add()([dense2, layers.Dense(64)(combined_features)])

    dense3 = layers.Dense(32, activation='gelu')(dense2_skip)
    dense3 = layers.BatchNormalization()(dense3)
    dense3 = layers.Dropout(0.2)(dense3)

    output = layers.Dense(9, activation='softmax')(dense3)

    return Model(inputs=inputs, outputs=output)

def feature_net_2():
    # Input: (sequence_length=2040, features=2)
    inputs = layers.Input(shape=(12240, 1), name='signal_input')

    # Encoder - Conv1D layers
    x = layers.Conv1D(32, 9, padding='same')(inputs)
    x = layers.Conv1D(32, 9, padding='same')(x)
    x = layers.Conv1D(32, 9, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(tf.nn.gelu)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.MaxPooling1D(2, padding='same')(x)

    # Modulation layers
    x = layers.Conv1D(40, 10, padding='same')(x)
    x = layers.Activation(tf.nn.gelu)(x)

    x = layers.Conv1D(48, 15, padding='same')(x)
    x = layers.Activation(tf.nn.gelu)(x)

    x = layers.Conv1D(48, 20, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2, padding='same')(x)
    x = layers.Activation(tf.nn.gelu)(x)

    # Dilated convolutions for larger receptive field
    x1 = layers.Conv1D(48, 20, padding='same', dilation_rate=1)(x)
    x2 = layers.Conv1D(48, 25, padding='same', dilation_rate=2)(x)
    x3 = layers.Conv1D(48, 30, padding='same', dilation_rate=4)(x)

    dilated_combined = layers.Concatenate()([x, x1, x2, x3])
    dilated_combined = layers.Dropout(0.2)(dilated_combined)

    # Attention mechanism for important feature selection
    attention = layers.Conv1D(192, 1, activation='sigmoid')(dilated_combined)
    attended_features = layers.Multiply()([dilated_combined, attention])
    attended_features = layers.Dropout(0.2)(attended_features)

    x = layers.GlobalAveragePooling1D()(attended_features)

    # Dense layers
    x = layers.Dense(128, activation='gelu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(64, activation='gelu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(32, activation='gelu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    # Output (18 classes)
    output = layers.Dense(9, activation='softmax')(x)

    return Model(inputs=inputs, outputs=output)


def build_trainable_voter(model1, model2, num_classes=9, train_base=False):
    model1._name = "pretrained_model1"
    model2._name = "pretrained_model2"
    # Base input (raw data shape)
    base_input = layers.Input(shape=(4080, 2), name="signal_input")

    # Adapt input to each model’s expected shape
    input_shape_1 = model1.input_shape[1:]
    input_shape_2 = model2.input_shape[1:]

    reshaped1 = layers.Reshape(input_shape_1)(base_input)
    reshaped2 = layers.Reshape(input_shape_2)(base_input)

    # Freeze or allow training
    model1.trainable = train_base
    model2.trainable = train_base

    # Get model outputs
    out1 = model1(reshaped1)
    out2 = model2(reshaped2)

    # Concatenate model outputs
    combined = layers.Concatenate()([out1, out2])  # e.g., (batch, 18)

    # Meta-learner (dense voter)
    x = layers.Dense(256, activation="relu")(combined)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    # Final ensemble model
    ensemble_model = Model(inputs=base_input, outputs=outputs, name="trainable_voter")

    return ensemble_model

def simple_cnn():
    inputs = layers.Input(shape=(12240, 1), name='signal_input')

    x = layers.Reshape((120, 102, 1), name='reshape_4d')(inputs)
    x = layers.Conv2D(16, (6, 2), activation='relu', padding='same', data_format='channels_first')(x)
    x = layers.BatchNormalization(axis=1)(x)
    x = layers.MaxPooling2D((3, 1), padding='same', data_format='channels_first')(x)

    x = layers.Conv2D(32, (9, 3), activation='relu', padding='same', data_format='channels_first')(x)
    x = layers.BatchNormalization(axis=1)(x)
    x = layers.MaxPooling2D((3, 1), padding='same', data_format='channels_first')(x)

    x = layers.Conv2D(64, (12, 4), activation='relu', padding='same', data_format='channels_first')(x)
    x = layers.BatchNormalization(axis=1)(x)
    x = layers.MaxPooling2D((3, 1), padding='same', data_format='channels_first')(x)

    x = layers.Conv2D(128, (15, 5), activation='relu', padding='same', data_format='channels_first')(x)
    x = layers.BatchNormalization(axis=1)(x)
    x = layers.MaxPooling2D((3, 1), padding='same', data_format='channels_first')(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    output = layers.Dense(9, activation='softmax')(x)

    return Model(inputs=inputs, outputs=output)


def cnn9():
    d = 0.3
    inputs = layers.Input(shape=(4080, 2, 1), name='signal_input')

    x = layers.Conv2D(64, (128, 1), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Dropout(d)(x)
    # x = layers.MaxPooling2D((2, 1), padding='same')(x)
    x = layers.Conv2D(32, (128, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Dropout(d)(x)
    # x = layers.MaxPooling2D((2, 1), padding='same')(x)
    x = layers.Conv2D(24, (128, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Dropout(d)(x)
    # x = layers.MaxPooling2D((1, 2), padding='same')(x)
    x = layers.Conv2D(16, (128, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Dropout(d)(x)
    # x = layers.MaxPooling2D((1, 2), padding='same')(x)
    x = layers.Conv2D(8, (128, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Dropout(d)(x)
    # x = layers.MaxPooling2D((1, 2), padding='same')(x)
    x = layers.Conv2D(16, (128, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Dropout(d)(x)
    # x = layers.MaxPooling2D((1, 2), padding='same')(x)
    x = layers.Conv2D(24, (128, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Dropout(d)(x)
    # x = layers.MaxPooling2D((1, 2), padding='same')(x)
    x = layers.Conv2D(32, (128, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Dropout(d)(x)
    # x = layers.MaxPooling2D((1, 2), padding='same')(x)
    x = layers.Conv2D(64, (128, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Dropout(d)(x)
    x = layers.MaxPooling2D((5, 1), padding='same')(x)
    x = layers.Flatten()(x)

    x = layers.Dense(80)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(40)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.BatchNormalization()(x)

    output = layers.Dense(9, activation='softmax')(x)

    return Model(inputs=inputs, outputs=output)

def feature_net():
    inputs = layers.Input(shape=(2040, 2, 1), name='signal_input')
    x = layers.Reshape((2040, 1, 2))(inputs)

    # encoder
    x = layers.Conv2D(32, (3, 1), padding='same')(x)
    x = layers.Conv2D(32, (7, 1), padding='same')(x)
    x = layers.Conv2D(32, (11, 1), padding='same')(x)
    x = layers.Conv2D(32, (15, 1), padding='same')(x)
    x = layers.AveragePooling2D((3, 3), padding='same')(x)

    # modulation

    # layer 56 is added and 32 -> 40
    x = layers.Conv2D(32, (20, 1), strides=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(64, (30, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(64, (40, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPooling2D((3, 3), padding='same')(x)
    x = layers.Dropout(0.3)(x)

    #interleaver
    dil1 = layers.Conv2D(32, (30, 1), padding='same', dilation_rate=(1, 1), activation='relu')(x)
    dil1 = layers.BatchNormalization()(dil1)

    dil2 = layers.Conv2D(32, (45, 1), padding='same', dilation_rate=(2, 1), activation='relu')(x)
    dil2 = layers.BatchNormalization()(dil2)

    dil3 = layers.Conv2D(32, (60, 1), padding='same', dilation_rate=(4, 1), activation='relu')(x)
    dil3 = layers.BatchNormalization()(dil3)

    # Combine dilated features
    dilated_combined = layers.Concatenate()([dil1, dil2, dil3])
    dilated_combined = layers.Dropout(0.3)(dilated_combined)

    # Attention mechanism for important feature selection
    attention = layers.Conv2D(96, (1, 1), activation='sigmoid')(dilated_combined)
    attended_features = layers.Multiply()([dilated_combined, attention])

    x = layers.Flatten()(attended_features)

    x = layers.Dense(96, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(48, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    output = layers.Dense(18, activation='softmax')(x)

    return Model(inputs=inputs, outputs=output)

def residual_tcn_block(x, filters, kernel_size=7, dilation_rate=1, dropout=0.1, name=None):
    """
    1D residual block with dilation (TCN-style):
    BN -> GELU -> Conv1D(dilated) -> BN -> GELU -> Dropout -> Conv1D(1x1) -> Skip add
    """
    shortcut = x
    inp_channels = x.shape[-1]
    x = layers.BatchNormalization(name=None if not name else f"{name}_bn1")(x)
    x = layers.Activation("gelu", name=None if not name else f"{name}_gelu1")(x)
    x = layers.Conv1D(filters, kernel_size, padding="same",
                      dilation_rate=dilation_rate,
                      name=None if not name else f"{name}_conv_dil")(x)

    x = layers.BatchNormalization(name=None if not name else f"{name}_bn2")(x)
    x = layers.Activation("gelu", name=None if not name else f"{name}_gelu2")(x)
    x = layers.Dropout(dropout, name=None if not name else f"{name}_drop")(x)
    x = layers.Conv1D(filters, 1, padding="same",
                      name=None if not name else f"{name}_conv1x1")(x)

    # Projection if channels mismatch
    if inp_channels != filters:
        shortcut = layers.Conv1D(filters, 1, padding="same",
                                 name=None if not name else f"{name}_proj")(shortcut)

    out = layers.Add(name=None if not name else f"{name}_add")([shortcut, x])
    return out

def transformer_block(x, num_heads=4, key_dim=32, dropout=0.1, name=None):
    attn_in = layers.LayerNormalization(name=None if not name else f"{name}_ln1")(x)
    attn_out = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim,
                                         dropout=dropout,
                                         name=None if not name else f"{name}_mha")(attn_in, attn_in)
    x = layers.Add(name=None if not name else f"{name}_mha_add")([x, attn_out])

    # FFN
    ffn_in = layers.LayerNormalization(name=None if not name else f"{name}_ln2")(x)
    f = layers.Dense(4 * x.shape[-1], activation="gelu",
                     name=None if not name else f"{name}_ffn1")(ffn_in)
    f = layers.Dropout(dropout, name=None if not name else f"{name}_ffn_drop")(f)
    f = layers.Dense(x.shape[-1], name=None if not name else f"{name}_ffn2")(f)
    x = layers.Add(name=None if not name else f"{name}_ffn_add")([x, f])
    return x

def build_rf_multitask_model(seq_len=2040, n_features=2, base_filters=64, kernel_size=7, dilations=(1, 2, 4, 8), attn_heads=4, attn_key_dim=32, dropout=0.1):
    # inputs = layers.Input(shape=(seq_len, n_features), name="signal_input")
    #
    # # Stem: small receptive field to begin
    # x = layers.Conv1D(base_filters, kernel_size, padding="same", name="stem_conv")(inputs)
    # x = layers.BatchNormalization(name="stem_bn")(x)
    # x = layers.Activation("gelu", name="stem_gelu")(x)
    #
    # # Stacked dilated residual blocks (TCN-style)
    # filters = base_filters
    # for i, d in enumerate(dilations):
    #     x = residual_tcn_block(x, filters=filters, kernel_size=kernel_size,
    #                            dilation_rate=d, dropout=dropout,
    #                            name=f"tcn{i+1}")
    #     # Optional downsample (temporal) to control length & enlarge receptive field
    #     if i < len(dilations) - 1:
    #         x = layers.MaxPooling1D(pool_size=2, padding="same", name=f"pool{i+1}")(x)
    #         # modestly increase channels
    #         filters = min(filters * 2, 256)
    #
    # # Lightweight self-attention on top of conv features
    # x = transformer_block(x, num_heads=attn_heads, key_dim=attn_key_dim,
    #                       dropout=dropout, name="xformer")
    #
    # # Global context
    # x = layers.GlobalAveragePooling1D(name="gap")(x)
    inputs = layers.Input(shape=(2040, 2), name='signal_input')  # I and Q channels as features

    # Encoder - Conv1D layers
    x = layers.Conv1D(64, 15, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(128, 9, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x5 = layers.Conv1D(128, 15, strides=1, padding='same')(x)
    x5 = layers.BatchNormalization()(x5)
    # x = layers.Activation(tf.nn.gelu)(x)
    # x = layers.Dropout(0.3)(x)

    x6 = layers.Conv1D(128, 20, padding='same')(x)
    x6 = layers.BatchNormalization()(x6)
    # x = layers.Activation(tf.nn.gelu)(x)
    # x = layers.Dropout(0.3)(x)

    x7 = layers.Conv1D(128, 45, padding='same', dilation_rate=1)(x)
    x7 = layers.BatchNormalization()(x7)
    # x = layers.Activation(tf.nn.gelu)(x)

    x8 = layers.Conv1D(128, 60, padding='same', dilation_rate=2)(x)
    x8 = layers.BatchNormalization()(x8)
    # x = layers.Activation(tf.nn.gelu)(x)
    dilated_combined = layers.Concatenate()([x5, x6, x7, x8])
    dilated_combined = layers.Dropout(0.3)(dilated_combined)

    # Attention mechanism for important feature selection
    attention = layers.Conv1D(512, 1, activation='sigmoid')(dilated_combined)
    attended_features = layers.Multiply()([dilated_combined, attention])

    x = layers.GlobalAveragePooling1D()(attended_features)
    x = layers.Dropout(dropout, name="prehead_drop")(x)

    # A small shared bottleneck
    shared = layers.Dense(256, activation="gelu", name="shared_dense")(x)
    shared = layers.BatchNormalization(name="shared_bn")(shared)
    shared = layers.Dropout(dropout, name="shared_drop")(shared)

    # ---- Heads ----
    # 1) Encoder: conv / turbo / ldpc (3 classes)
    enc = layers.Reshape((-1, 1))(shared)  # fake seq for conv
    enc = layers.Conv1D(128, 5, activation="gelu", padding="same")(enc)
    enc = residual_tcn_block(enc, 128)
    enc = residual_tcn_block(enc, 128, dilation_rate=2)  # dilated conv to expand view
    # Attention to boost representation
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=32)(enc, enc)
    enc = layers.Add()([enc, attn])
    enc = layers.GlobalAveragePooling1D()(enc)
    encoder_out = layers.Dense(3,
                               activation="softmax",
                               name="encoder_out")(enc)

    # # 2) Modulation: 8psk / 16qam / 64qam (3 classes)
    # mod1 = layers.Dense(96, activation="gelu", name="mod_fc1")(shared)
    # mod1 = layers.Dropout(dropout, name="mod_drop1")(mod1)
    # mod2 = layers.Dense(64, activation="gelu", name="mod_fc2")(mod1)
    # mod2 = layers.Dropout(dropout, name="mod_drop2")(mod2)
    # modulation_out = layers.Dense(3, activation="softmax", name="modulation_out")(mod2)

    # 3) Interleaver: block / conv (2 classes)
    intl = layers.Reshape((-1, 1))(shared)
    intl = layers.Conv1D(128, 7, activation="gelu", padding="same")(intl)
    intl = residual_tcn_block(intl, 64)
    intl = residual_tcn_block(intl, 64, dilation_rate=3)
    intl = layers.GlobalAveragePooling1D()(intl)
    interleaver_out = layers.Dense(2,
                                   activation="softmax",
                                   name="interleaver_out")(intl)

    model = Model(inputs=inputs,
                  outputs=[encoder_out, #modulation_out,
                           interleaver_out],
                  name="RF_MTL_TCN_Attn")
    return model
