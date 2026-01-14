import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Supress Tensorflow info/warning messages

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from sklearn.model_selection import train_test_split

# Load MNIST dataset
(x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess the Images
# Normalize pixel values to the range [0, 1]
x_train_full = x_train_full / 255.0
x_test = x_test / 255.0

# Split dataset into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(
    x_train_full,
    y_train_full,
    test_size=0.2,
    random_state=42
)

# Define the neural network architecture
# Create a sequential model with Flatten & two Dense layers
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=5,
    batch_size=32
)

# Evaluate the model on the Test Set
test_loss, test_accuracy = model.evaluate(x_test, y_test)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)