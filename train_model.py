import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

# Define a simple CNN model
def build_cnn_model():
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Ensure model directory exists
model_dir = "model"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Train a new model (Dummy training data)
X_train = np.random.rand(100, 128, 128, 1)  # 100 fake grayscale images
y_train = np.random.randint(0, 2, 100)  # Random labels (0 or 1)

model = build_cnn_model()
model.fit(X_train, y_train, epochs=5, batch_size=16, verbose=1)  # Train for 5 epochs

# Save model
model.save("model/medical_model.h5")
print("Model saved successfully!")
