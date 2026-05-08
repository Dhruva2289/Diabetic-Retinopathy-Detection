import os
import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load trained model
model_path = os.path.join(os.path.dirname(__file__), "model.h5")
model = None

if os.path.exists(model_path):
    model = load_model(model_path)
else:
    st.warning(
        "Trained model file not found. Please train the model and save it as `model.h5` in this folder "
        "or run a training script that writes `model.h5` before using this app."
    )

# Labels
classes = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR"
}

# Page title
st.title("🩺 Diabetic Retinopathy Detection")

st.write("Upload a retina image and the AI model will predict severity level.")

# Upload image
uploaded_file = st.file_uploader(
    "Choose Retina Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file)

    # Display image
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert image to numpy array
    img = np.array(image)

    # Resize image
    img = cv2.resize(img, (64 , 64))

    # Normalize image
    img = img / 255.0

    # Reshape image
    img = np.reshape(img, (1, 64, 64, 3))

    # Prediction button
    if model is None:
        st.error(
            "No trained model is loaded. Please place `model.h5` in this folder or train the model first."
        )
    else:
        if st.button("Predict"):
            prediction = model.predict(img)
            result = np.argmax(prediction)
            st.success(f"Prediction: {classes[result]}")