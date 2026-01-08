import streamlit as st

# PAGE CONFIG 
st.set_page_config(
    page_title="Smart Waste Classifier",
    layout="centered"
)

import tensorflow as tf
import numpy as np
from PIL import Image

# KONFIGURASI
IMG_SIZE = (224, 224)

GENERAL_CLASSES = [
    'battery', 'cardboard', 'glass', 'metal',
    'organic', 'paper', 'plastic', 'textile', 'trash'
]

PLASTIC_CLASSES = ['HDPE', 'LDPA', 'Other', 'PET', 'PP', 'PS', 'PVC']

# LOAD MODEL 
@st.cache_resource
def load_models():
    try:
        general_model = tf.keras.models.load_model(
            "models/efficientnet_b0_best.h5",
            compile=False
        )
        plastic_model = tf.keras.models.load_model(
            "models/modelb0_best.h5",
            compile=False
        )
        return general_model, plastic_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

general_model, plastic_model = load_models()

# PREPROCESS IMAGE
def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    image = np.array(image)
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    return np.expand_dims(image, axis=0)

def center_crop_square(img):
    w, h = img.size
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim
    return img.crop((left, top, right, bottom))

# STREAMLIT UI
st.title("♻️ Smart Waste Classification System")

uploaded_file = st.file_uploader(
    "Upload waste image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    
    display_image = center_crop_square(image).resize((300, 300))
    st.image(display_image, caption="Image Input")
    

if st.button("Predict", type="primary"):

    if general_model is None or plastic_model is None:
        st.error("Model not found")
    else:
        status = st.empty()
        progress_bar = st.progress(0)

        status.info("📤 Image received")
        progress_bar.progress(20)

        status.info("⚙️ Preprocessing image...")
        input_tensor = preprocess_image(image)
        progress_bar.progress(40)

        status.info("🔍 Identifying waste")
        general_pred = general_model.predict(input_tensor)
        general_idx = np.argmax(general_pred)
        general_label = GENERAL_CLASSES[general_idx]
        general_conf = np.max(general_pred) * 100
        progress_bar.progress(65)

        st.divider()
        st.subheader("🔍 Classification Result")
        st.write(f"**Main Category:** {general_label.upper()}")
        st.progress(int(general_conf), text=f"**Confidence:** {general_conf:.2f}%")

        if general_label == "organic":
            status.success("🌱 Organic waste detected")
            progress_bar.progress(100)

            st.success("🌱 Organic Waste Detected")
            st.info("Can be used for fertilizer / compost")

        else:
            status.info("🧪 Anorganic waste detected")
            progress_bar.progress(80)

            st.warning("🧪 Anorganic Waste")

            if general_label == "plastic":
                status.info("🔬 Identifying plastic type...")
                plastic_pred = plastic_model.predict(input_tensor)
                plastic_idx = np.argmax(plastic_pred)
                plastic_label = PLASTIC_CLASSES[plastic_idx]
                plastic_conf = np.max(plastic_pred) * 100
                progress_bar.progress(95)

                st.subheader("🔬 Plastic Type Details")
                st.success(f"♻️ Plastic Type: **{plastic_label}**")
                st.progress(int(plastic_conf), text=f"**Confidence:** {plastic_conf:.2f}%")
                status.success(f"✅ Plastic identified as {plastic_label}")

                progress_bar.progress(100)

            elif general_label in ['glass', 'metal', 'paper', 'cardboard']:
                progress_bar.progress(100)
                st.info("💡 Make sure the trash is clean before recycling.")


