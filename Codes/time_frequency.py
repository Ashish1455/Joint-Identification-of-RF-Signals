import tensorflow as tf
from tensorflow import keras

# Load the model
model = keras.models.load_model('EM pictures/Feature_net8680.h5')

# Print model summary
print(model.summary())

# Get detailed architecture
print("\nModel Config:")
print(model.get_config())