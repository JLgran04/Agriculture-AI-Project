import os
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import google.generativeai as genai
from dotenv import load_dotenv

# ---------------------------
# Setup
# ---------------------------
st.set_page_config(page_title="Smart Agriculture AI", page_icon="🌿", layout="wide")

# Load .env locally (no effect on Streamlit Cloud unless you add a .env file;
# on Cloud use Settings → Secrets)
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
else:
    gemini_model = None  # we'll handle this gracefully below

# ---------------------------
# Load Models
# ---------------------------
# Ensure your repo has: models/soil_moisture_model.keras and models/plant_disease_model.keras
soil_model = load_model("models/soil_moisture_model.keras")
plant_model = load_model("models/plant_disease_model.keras")

# ---------------------------
# Class Labels
# ---------------------------
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

# ---------------------------
# Image Preprocessing (PIL + NumPy; no keras.preprocessing)
# ---------------------------
def preprocess_image(img: Image.Image, target_size=(150, 150)):
    img = img.resize(target_size)
    arr = np.array(img).astype("float32") / 255.0   # HWC, [0..1]
    arr = np.expand_dims(arr, axis=0)               # NHWC
    return arr

# ---------------------------
# Prediction Functions
# ---------------------------
def predict_soil(img: Image.Image):
    preds = soil_model.predict(preprocess_image(img))
    idx = int(np.argmax(preds[0]))
    prob = float(preds[0][idx])
    return soil_class_labels.get(idx, "Unknown"), prob

def predict_plant(img: Image.Image):
    preds = plant_model.predict(preprocess_image(img))
    idx = int(np.argmax(preds[0]))
    prob = float(preds[0][idx])
    return plant_class_labels.get(idx, "Unknown"), prob

# ---------------------------
# Gemini Explanation
# ---------------------------
def explain_prediction(label: str, category: str) -> str:
    if not gemini_model:
        return ("Gemini key not found. Add GEMINI_API_KEY in Streamlit Secrets "
                "to enable natural-language advice.")
    prompt = (
        f"You are an agriculture expert. Explain in clear, farmer-friendly language "
        f"what it means when the {category} prediction is \"{label}\". "
        f"Include practical, step-by-step advice."
    )
    try:
        resp = gemini_model.generate_content(prompt)
        return (resp.text or "").strip()
    except Exception as e:
        return f"Gemini explanation unavailable right now: {e}"

# ---------------------------
# UI
# ---------------------------
st.markdown("<h1 style='text-align:center; color:#4CAF50;'>🌱 Smart Agriculture AI</h1>",
            unsafe_allow_html=True)
st.write("### Analyze soil moisture and detect plant diseases with AI, plus Gemini advice.")

col1, col2 = st.columns(2, gap="large")

with col1:
    choice = st.radio("Input method:", ("📁 Upload Image", "📸 Use Camera"))
    img = None
    if choice == "📁 Upload Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            img = Image.open(uploaded_file).convert("RGB")
    else:
        cam_img = st.camera_input("Take a photo")
        if cam_img:
            img = Image.open(cam_img).convert("RGB")

with col2:
    if img is not None:
        st.image(img, caption="Input Image", use_column_width=True)

st.markdown("---")

if img is not None:
    category = st.radio("Select Prediction Type:", ["🌍 Soil Moisture", "🌾 Plant Disease"])
    if st.button("🔍 Analyze"):
        with st.spinner("Analyzing..."):
            if category == "🌍 Soil Moisture":
                label, prob = predict_soil(img)
                explanation = explain_prediction(label, "soil moisture")
            else:
                label, prob = predict_plant(img)
                explanation = explain_prediction(label, "plant disease")

            st.success(f"### ✅ Prediction: **{label}** (Confidence: {prob:.2f})")
            st.info(f"💬 **Advice:**\n\n{explanation}")
else:
    st.warning("Please upload or capture an image to continue.")

