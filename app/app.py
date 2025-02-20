import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Set Streamlit page configuration
st.set_page_config(page_title="Pneumonia Detection App", page_icon="🩺")

# Load the model
model_path = os.path.join(os.getcwd(), "saved_model", "model_VGG16.keras")
model = load_model(model_path)

# Define image size (use the same size as your model input)
IMG_SIZE = (224, 224)  # Adjust this based on your model input size

st.title("Pneumonia Detection from Chest X-Ray")

uploaded_file = st.file_uploader("Upload a Chest X-Ray image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image
    try:
        # Convert the uploaded file to a format compatible with Keras
        img = image.load_img(uploaded_file, target_size=IMG_SIZE, color_mode='rgb')  # RGB for VGG16
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Model expects a batch
        img_array = img_array / 255.0  # Rescale to [0, 1]

        # Debugging: Check the shape before prediction
        st.write(f"Image shape before prediction: {img_array.shape}")

        # Make prediction
        prediction = model.predict(img_array)

        # Assuming binary classification: 0 -> Normal, 1 -> Pneumonia
        if prediction[0][0] > 0.5:
            st.error("Prediction: Pneumonia")
        else:
            st.success("Prediction: Normal")

    except Exception as e:
        st.error(f"Error in processing the image: {e}")

st.text("This model is based on VGG16 for Pneumonia Detection.")
