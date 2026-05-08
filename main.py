import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense
import matplotlib.pyplot as plt

# Use script-relative paths so the CSV and image folder load correctly
test_dir = os.path.dirname(os.path.abspath(__file__))

# Load CSV
data = pd.read_csv(os.path.join(test_dir, "train.csv"))

images = []
labels = []

# Read first 500 images only (for beginner testing)
for index, row in data.head(500).iterrows():

    image_path = os.path.join(test_dir, "train_images", row["id_code"] + ".png")

    image = cv2.imread(image_path)

    if image is not None:

        image = cv2.resize(image, (128, 128))

        images.append(image)

        labels.append(row["diagnosis"])

# Convert to numpy arrays
X = np.array(images) / 255.0
y = np.array(labels)

# Convert labels to categorical
y = to_categorical(y, 5)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build CNN model
model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu',
                 input_shape=(128,128,3)))

model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (3,3), activation='relu'))

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

# Train model
history = model.fit(
    X_train,
    y_train,
    epochs=5,
    validation_data=(X_test, y_test)
)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)

print("Accuracy:", accuracy)

# Save trained model for app.py
model.save(os.path.join(test_dir, "model.h5"))
print("Saved model to model.h5")

# Plot accuracy graph
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'])

plt.show()