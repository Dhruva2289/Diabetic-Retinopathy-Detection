import os
import cv2
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense

# Load CSV file
data = pd.read_csv("train.csv")

images = []
labels = []

print("Loading images...")

# Load first 500 images for beginner training
for index, row in data.head(500).iterrows():

    image_path = "train_images/" + row["id_code"] + ".png"

    image = cv2.imread(image_path)

    if image is not None:

        # Resize image
        image = cv2.resize(image, (64, 64))

        images.append(image)

        labels.append(row["diagnosis"])

print("Images loaded successfully!")

# Convert to numpy arrays
X = np.array(images) / 255.0
y = np.array(labels)

# Convert labels to categorical
y = to_categorical(y, 5)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Build CNN model
model = Sequential()

model.add(Conv2D(
    32,
    (3,3),
    activation='relu',
    input_shape=(64,64,3)
))

model.add(MaxPooling2D((2,2)))

model.add(Conv2D(
    64,
    (3,3),
    activation='relu'
))

model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dense(5, activation='softmax'))

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Training model...")

# Train model
model.fit(
    X_train,
    y_train,
    epochs=5,
    validation_data=(X_test, y_test)
)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)

print("Accuracy:", accuracy)

# Save trained model
model.save("model.h5")

print("Model saved as model.h5")