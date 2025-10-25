import os
import streamlit as st
import numpy as np
from PIL import Image
import keras
import google.generativeai as genai
from dotenv import load_dotenv

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(page_title="Smart Agriculture AI", page_icon="🌿", layout="wide")

st.markdown(
    "<h1 style='text-align:center; color:#4CAF50;'>🌱 Smart Agriculture AI</h1>",
    unsafe_allow_html=True
)
st.write("### Analyze soil moisture and detect plant diseases with AI, plus Gemini advice.")

# -------------------------------------------------
# Environment / API key (Gemini)
# -------------------------------------------------
# On Streamlit Cloud: set GEMINI_API_KEY in Secrets as:
# GEMINI_API_KEY = "your_real_key"
# Locally: put it in a .env file
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    # Use a currently exposed public model name
    gemini_model = genai.GenerativeModel("gemini-flash-latest")
else:
    gemini_model = None

# -------------------------------------------------
# Load Models (with graceful fallback)
# -------------------------------------------------
soil_model = None
plant_model = None
soil_model_error = None
plant_model_error = None

try:
    soil_model = keras.models.load_model("models/soil_moisture_model.keras")
except Exception as e:
    soil_model_error = str(e)

try:
    plant_model = keras.models.load_model("models/plant_disease_model.keras")
except Exception as e:
    plant_model_error = str(e)

# -------------------------------------------------
# Class Labels
# -------------------------------------------------
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
    # if your model has index 9, add it here:
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

# -------------------------------------------------
# Image Preprocessing (PIL + NumPy)
# -------------------------------------------------
def preprocess_image(img: Image.Image, target_size=(150, 150)):
    """
    Resize to model input size, normalize to [0,1], add batch dimension.
    Returns shape (1, H, W, C)
    """
    img = img.resize(target_size)
    arr = np.array(img).astype("float32") / 255.0  # HWC
    arr = np.expand_dims(arr, axis=0)              # 1,H,W,C
    return arr

# -------------------------------------------------
# Prediction helpers
# -------------------------------------------------
def predict_soil(img: Image.Image):
    """
    Returns (label, confidence)
    If soil_model didn't load, returns an error label instead of crashing.
    """
    if soil_model is None:
        return f"[Soil model not loaded: {soil_model_error}]", 0.0

    preds = soil_model.predict(preprocess_image(img))
    idx = int(np.argmax(preds[0]))
    prob = float(preds[0][idx])
    label = soil_class_labels.get(idx, "Unknown")
    return label, prob

def predict_plant(img: Image.Image):
    """
    Returns (label, confidence)
    If plant_model didn't load, returns an error label instead of crashing.
    """
    if plant_model is None:
        return f"[Plant model not loaded: {plant_model_error}]", 0.0

    preds = plant_model.predict(preprocess_image(img))
    idx = int(np.argmax(preds[0]))
    prob = float(preds[0][idx])
    label = plant_class_labels.get(idx, "Unknown")
    return label, prob

# -------------------------------------------------
# Gemini Explanation
# -------------------------------------------------
def explain_prediction(label: str, category: str) -> str:
    """
    Uses Gemini to generate farmer-friendly advice.
    Falls back to a helpful message if no API key.
    """
    if not gemini_model:
        return (
            "Natural-language advice is disabled because no Gemini API key was found. "
            "Add GEMINI_API_KEY in Streamlit Secrets to enable this."
        )

   prompt = (
    f"You are an agriculture expert who speaks both English and Arabic fluently. "
    f"Explain in Arabic what it means when the {category} prediction is \"{label}\". "
    f"Write the explanation in Modern Standard Arabic, suitable for farmers. "
    f"Include short practical advice or steps to follow."
    )


    try:
        resp = gemini_model.generate_content(prompt)
        text = (resp.text or "").strip()
        if not text:
            return "No explanation generated."
        return text
    except Exception as e:
        return f"Gemini explanation unavailable right now: {e}"

# -------------------------------------------------
# UI Layout
# -------------------------------------------------
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("📷 Input")
    choice = st.radio("Input method:", ("📁 Upload Image", "📸 Use Camera"))
    img = None

    if choice == "📁 Upload Image":
        uploaded_file = st.file_uploader(
            "Upload a soil or plant image",
            type=["jpg", "jpeg", "png"]
        )
        if uploaded_file:
            img = Image.open(uploaded_file).convert("RGB")
    else:
        cam_img = st.camera_input("Take a live photo")
        if cam_img:
            img = Image.open(cam_img).convert("RGB")

with col2:
    st.subheader("👀 Preview")
    if img is not None:
        st.image(img, caption="Input Image", use_column_width=True)
    else:
        st.info("No image yet. Upload or take a photo to preview it here.")

st.markdown("---")

# Bottom interaction area
if img is not None:
    st.subheader("🔬 Analysis")
    category = st.radio("Select Prediction Type:", ["🌍 Soil Moisture", "🌾 Plant Disease"])

    if st.button("🔍 Analyze"):
        with st.spinner("Analyzing image..."):
            # run prediction
            if category == "🌍 Soil Moisture":
                label, prob = predict_soil(img)
                explanation = explain_prediction(label, "soil moisture")
            else:
                label, prob = predict_plant(img)
                explanation = explain_prediction(label, "plant disease")

            # show results
            st.success(f"### ✅ Prediction: **{label}** (Confidence: {prob:.2f})")
            st.info(f"💬 **Advice:**\n\n{explanation}")

else:
    st.warning("Please upload or capture an image to continue.")






