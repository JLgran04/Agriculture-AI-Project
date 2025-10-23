import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv(AIzaSyCIsD_AEfWIvfSI5zR3Uv29e0czSAkeNdI))

# Load Models
soil_model = load_model("models/soil_moisture_model.keras")
plant_model = load_model("models/plant_disease_model.keras")

# Class Labels
soil_class_labels = {0: "dry", 1: "moist", 2: "wet"}
plant_class_labels = {
    0: "Corn (Cercospora leaf spot - Gray leaf spot)",
    1: "Corn (Common rust)",
    2: "Corn (Northern Leaf Blight)",
    3: "Corn (Healthy)",
    4: "Pepper (Bacterial spot)",
    5: "Pepper (Healthy)",
    6: "Potato (Early blight)",
    7: "Potato (Late blight)",
    8: "Potato (Healthy)",
    10: "Strawberry (Leaf scorch)",
    11: "Strawberry (Healthy)",
    12: "Tomato (Bacterial spot)",
    13: "Tomato (Early blight)",
    14: "Tomato (Late blight)",
    15: "Tomato (Leaf Mold)",
    16: "Tomato (Septoria leaf spot)",
    17: "Tomato (Spider mites / Two-spotted spider mite)",
    18: "Tomato (Target Spot)",
    19: "Tomato (Yellow Leaf Curl Virus)",
    20: "Tomato (Mosaic virus)",
    21: "Tomato (Healthy)"
}

# Image Preprocessing
def preprocess_image(img: Image.Image, target_size=(150,150)):
    img = img.resize(target_size)
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# Prediction Functions
def predict_soil(img):
    preds = soil_model.predict(preprocess_image(img))
    i = np.argmax(preds[0])
    return soil_class_labels[i], preds[0][i]

def predict_plant(img):
    preds = plant_model.predict(preprocess_image(img))
    i = np.argmax(preds[0])
    return plant_class_labels[i], preds[0][i]

# Gemini Explanation
def explain_prediction(label, category):
    prompt = f"""
    You are an agriculture expert. Explain in simple, human-friendly language what it means when
    the {category} prediction is "{label}". Include practical advice or interpretation for a farmer.
    """
    response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)
    return response.text.strip()

# --- Streamlit Interface ---
st.set_page_config(page_title="Smart Agriculture AI", page_icon="🌿", layout="wide")

st.markdown("<h1 style='text-align:center; color:#4CAF50;'>🌱 Smart Agriculture AI</h1>", unsafe_allow_html=True)
st.write("### Analyze soil moisture and detect plant diseases using Artificial Intelligence & Gemini AI explanations.")

col1, col2 = st.columns(2)
with col1:
    choice = st.radio("Input method:", ("📁 Upload Image", "📸 Use Camera"))
    img = None
    if choice == "📁 Upload Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            img = Image.open(uploaded_file).convert("RGB")
    else:
        cam_img = st.camera_input("Take a live photo")
        if cam_img:
            img = Image.open(cam_img).convert("RGB")

with col2:
    if img:
        st.image(img, caption="Input Image", use_column_width=True)

st.markdown("---")
if img:
    category = st.radio("Select Prediction Type:", ["🌍 Soil Moisture", "🌾 Plant Disease"])
    if st.button("🔍 Analyze"):
        with st.spinner("AI analyzing..."):
            if category == "🌍 Soil Moisture":
                label, prob = predict_soil(img)
                explanation = explain_prediction(label, "soil moisture")
            else:
                label, prob = predict_plant(img)
                explanation = explain_prediction(label, "plant disease")
            
            st.success(f"### ✅ Prediction: **{label}** (Confidence: {prob:.2f})")
            st.info(f"💬 **Gemini AI Explanation:**\n\n{explanation}")
else:
    st.warning("Please upload or capture an image to continue.")
