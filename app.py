import os
import streamlit as st
import numpy as np
from PIL import Image
import keras
import google.generativeai as genai
from dotenv import load_dotenv

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Smart Agriculture AI",
    page_icon="🌿",
    layout="wide"
)

# UI Design layout
st.markdown(
    """
    <style>
    body {
        background-color: #f7f8fa;
        font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto;
    }
    .main-header {
        background: linear-gradient(90deg, #2e7d32 0%, #66bb6a 100%);
        color: white;
        padding: 1.2rem 2rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.2);
        box-shadow: 0 12px 30px rgba(0,0,0,0.15);
    }
    .main-header h1 {
        font-size: 1.5rem;
        font-weight: 600;
        margin: 0;
        color: #fff;
    }
    .main-header p {
        font-size: 0.9rem;
        margin: 0.3rem 0 0 0;
        color: rgba(255,255,255,0.9);
    }

    .card {
        background: #ffffff;
        border-radius: 14px;
        padding: 1.2rem 1.2rem 1rem 1.2rem;
        border: 1px solid #e5e7eb;
        box-shadow: 0 6px 24px rgba(0,0,0,0.04);
    }

    .card h3 {
        font-size: 1rem;
        font-weight: 600;
        color: #2e7d32;
        margin-top: 0;
        margin-bottom: .75rem;
        display: flex;
        align-items: center;
        gap: .5rem;
    }

    .section-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: .5rem;
    }

    .result-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 1rem 1.2rem;
        border: 1px solid #d1d5db;
        box-shadow: 0 12px 30px rgba(0,0,0,0.06);
        margin-bottom: 1rem;
    }

    .result-label {
        font-weight: 600;
        font-size: 1.1rem;
        color: #2e7d32;
        margin-bottom: .25rem;
    }

    .confidence {
        font-size: 0.9rem;
        color: #4b5563;
        margin-bottom: 0.75rem;
    }

    .advice-wrapper {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 1rem 1rem;
        font-size: 0.9rem;
        color: #1f2937;
        line-height: 1.5rem;
        white-space: pre-wrap;
    }
    .advice-header {
        font-weight: 600;
        margin-bottom: .5rem;
        font-size: .95rem;
        color: #374151;
    }

    .rtl-block {
        direction: rtl;
        text-align: right;
        background: #fff;
        border-radius: 8px;
        border: 1px solid #d1d5db;
        padding: .75rem .8rem;
        margin-top: .75rem;
        font-size: 0.9rem;
        line-height: 1.5rem;
        color: #1f2937;
        box-shadow: 0 4px 16px rgba(0,0,0,0.03);
    }

    .analyze-button button {
        width: 100% !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        background: linear-gradient(90deg, #2e7d32 0%, #66bb6a 100%) !important;
        border: 0 !important;
        color: white !important;
        box-shadow: 0 10px 24px rgba(46,125,50,.4) !important;
    }

    .warning-box {
        background: #fff7ed;
        border: 1px solid #fdba74;
        color: #9a3412;
        border-radius: 10px;
        padding: .8rem 1rem;
        font-size: .9rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# Header
# -------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <h1>🌿 Smart Agriculture AI Assistant</h1>
        <p>Image-based soil moisture & plant disease analysis with bilingual field guidance.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# Environment / API Key
# -------------------------------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    # If this model gives 404, switch to "gemini-pro-latest"
    gemini_model = genai.GenerativeModel("gemini-flash-latest")
else:
    gemini_model = None

# -------------------------------------------------
# Load Models (safe fallback)
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
# Labels
# -------------------------------------------------
soil_class_labels = {
    0: "dry",
    1: "moist",
    2: "wet"
}

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

# -------------------------------------------------
# Image Preprocessing
# -------------------------------------------------
def preprocess_image(img: Image.Image, target_size=(150, 150)):
    img = img.resize(target_size)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

# -------------------------------------------------
# Prediction logic
# -------------------------------------------------
def predict_soil(img: Image.Image):
    if soil_model is None:
        return f"[Soil model not loaded: {soil_model_error}]", 0.0
    preds = soil_model.predict(preprocess_image(img))
    idx = int(np.argmax(preds[0]))
    prob = float(preds[0][idx])
    label = soil_class_labels.get(idx, "Unknown")
    return label, prob

def predict_plant(img: Image.Image):
    if plant_model is None:
        return f"[Plant model not loaded: {plant_model_error}]", 0.0
    preds = plant_model.predict(preprocess_image(img))
    idx = int(np.argmax(preds[0]))
    prob = float(preds[0][idx])
    label = plant_class_labels.get(idx, "Unknown")
    return label, prob

# -------------------------------------------------
# Gemini Advice (English + Arabic)
# -------------------------------------------------
def explain_prediction(label: str, category: str) -> str:
    if not gemini_model:
        return (
            "🌐 Gemini is not configured. Add your GEMINI_API_KEY in Streamlit Secrets."
        )

    prompt = (
        f"You are an experienced agricultural field advisor who helps farmers in real conditions. "
        f"The AI system predicted {category} = \"{label}\".\n\n"
        f"Your job:\n"
        f"1. Explain what this result means and why it matters.\n"
        f"2. Give clear, practical next steps the farmer should take in the next 24 hours.\n"
        f"3. Give prevention tips for the next few days.\n"
        f"4. If it is a disease, explain if the crop should be isolated, sprayed, pruned, or monitored.\n"
        f"5. If it is soil moisture, give watering guidance: how much, how often, and what to watch for.\n\n"
        f"Answer in TWO sections:\n\n"
        f"### English Explanation\n"
        f"- Write in simple English for a non-technical farmer.\n"
        f"- Use bullet points for actions.\n\n"
        f"### Arabic Explanation (الفهم بالعربية)\n"
        f"- اكتب شرحاً تفصيلياً باللغة العربية الفصحى السهلة.\n"
        f"- استخدم نقاط واضحة لخطوات العمل.\n"
        f"- اجعل النص عملي جداً (مثل: اسقِ التربة الآن / افحص الأوراق غداً / اعزل النبتة إذا كانت مصابة).\n"
    )

    try:
        resp = gemini_model.generate_content(prompt)
        text = (resp.text or "").strip()
        return text if text else "No explanation generated."
    except Exception as e:
        return f"Gemini explanation unavailable right now: {e}"

# -------------------------------------------------
# Layout: Input / Preview Columns
# -------------------------------------------------
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3>📥 Input Image</h3>', unsafe_allow_html=True)

    input_method = st.radio(
        "Choose how to provide an image:",
        ("Upload Image", "Use Camera")
    )

    img = None
    if input_method == "Upload Image":
        uploaded = st.file_uploader(
            "Upload a soil or plant image",
            type=["jpg", "jpeg", "png"]
        )
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
    else:
        cam_img = st.camera_input("Take a live photo")
        if cam_img:
            img = Image.open(cam_img).convert("RGB")

    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3>Preview & Task</h3>', unsafe_allow_html=True)

    if img is not None:
        st.image(img, caption="Preview", use_column_width=True)
    else:
        st.markdown(
            '<div class="warning-box">No image yet. Upload or take a photo.</div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="section-title">What do you want to analyze?</div>', unsafe_allow_html=True)
    task_type = st.radio(
        "",
        ["Soil Moisture", "Plant Disease"],
        horizontal=True
    )

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# -------------------------------------------------
# Analyze button row
# -------------------------------------------------
analyze_clicked = False
if img is not None:
    with st.container():
        st.markdown('<div class="analyze-button">', unsafe_allow_html=True)
        analyze_clicked = st.button("Analyze Image with AI")
        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown(
        '<div class="warning-box"> Please provide an image first to run analysis.</div>',
        unsafe_allow_html=True
    )

# -------------------------------------------------
# Results Section
# -------------------------------------------------
if analyze_clicked and img is not None:
    with st.spinner("Analyzing image and generating advice..."):
        if task_type == "Soil Moisture":
            label, prob = predict_soil(img)
            explanation_raw = explain_prediction(label, "soil moisture")
        else:
            label, prob = predict_plant(img)
            explanation_raw = explain_prediction(label, "plant disease")

    # Split English / Arabic sections for nicer display boxes
    english_part = ""
    arabic_part = ""

    # Heuristic split based on the headings we force in the prompt
    if "### Arabic Explanation" in explanation_raw:
        parts = explanation_raw.split("### Arabic Explanation")
        english_part = parts[0].replace("### English Explanation", "").strip()
        arabic_part = parts[1].strip()
    else:
        english_part = explanation_raw

    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="result-label">✅ Prediction: <span style="color:#000">{label}</span></div>
        <div class="confidence">Confidence: {prob:.2f}</div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="advice-wrapper">', unsafe_allow_html=True)
    st.markdown('<div class="advice-header">English Guidance</div>', unsafe_allow_html=True)
    st.markdown(english_part, unsafe_allow_html=False)
    st.markdown('</div>', unsafe_allow_html=True)

    if arabic_part:
        st.markdown('<div class="rtl-block">', unsafe_allow_html=True)
        st.markdown('<b>الإرشادات بالعربية</b><br>', unsafe_allow_html=True)
        st.markdown(arabic_part, unsafe_allow_html=False)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

