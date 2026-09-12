import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="NeuroScan | Brain Tumour Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# MODEL
# =========================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("brain_tumour_classifier.keras")


model = load_model()


# =========================================================
# CLASS NAMES
# =========================================================

class_names = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

display_names = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No Tumor",
    "pituitary": "Pituitary"
}


# =========================================================
# CSS
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(59,130,246,0.10), transparent 30%),
        radial-gradient(circle at 90% 20%, rgba(139,92,246,0.08), transparent 30%),
        #f8fafc;
}

.block-container {
    max-width: 1180px;
    padding-top: 1rem;
    padding-bottom: 3rem;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}


/* ================= NAVBAR ================= */

.navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 4px 25px 4px;
}

.logo {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 21px;
    font-weight: 800;
    color: #111827;
}

.logo-icon {
    width: 38px;
    height: 38px;
    border-radius: 11px;
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 20px;
    box-shadow: 0 8px 20px rgba(37,99,235,0.25);
}

.nav-badge {
    padding: 7px 13px;
    border-radius: 20px;
    background: #eef2ff;
    color: #4f46e5;
    font-size: 12px;
    font-weight: 600;
}


/* ================= HERO ================= */

.hero {
    text-align: center;
    padding: 35px 20px 45px;
}

.hero-tag {
    display: inline-block;
    padding: 7px 14px;
    border-radius: 20px;
    background: #eff6ff;
    color: #2563eb;
    font-size: 12px;
    font-weight: 700;
    margin-bottom: 16px;
}

.hero-title {
    font-size: 46px;
    line-height: 1.12;
    font-weight: 800;
    color: #111827;
    margin: 0;
}

.hero-highlight {
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-description {
    max-width: 650px;
    margin: 18px auto 0;
    color: #64748b;
    font-size: 16px;
    line-height: 1.7;
}


/* ================= SECTION TITLES ================= */

.section-title {
    text-align: center;
    font-size: 27px;
    font-weight: 800;
    color: #111827;
    margin: 15px 0 7px;
}

.section-subtitle {
    text-align: center;
    color: #64748b;
    font-size: 14px;
    margin-bottom: 25px;
}


/* ================= CARDS ================= */

.card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 20px;
    padding: 25px;
    box-shadow: 0 10px 35px rgba(15,23,42,0.05);
}

.upload-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 22px;
    padding: 30px;
    box-shadow: 0 15px 45px rgba(15,23,42,0.07);
}

.upload-title {
    text-align: center;
    font-size: 20px;
    font-weight: 700;
    color: #111827;
}

.upload-description {
    text-align: center;
    color: #64748b;
    font-size: 13px;
    margin-top: 6px;
}


/* ================= FILE UPLOADER ================= */

[data-testid="stFileUploader"] {
    background: #f8fafc;
    border: 2px dashed #cbd5e1;
    border-radius: 16px;
    padding: 12px;
}

[data-testid="stFileUploader"]:hover {
    border-color: #93c5fd;
}


/* ================= BUTTON ================= */

.stButton > button {
    width: 100%;
    min-height: 48px;
    border-radius: 12px;
    border: none;
    font-weight: 700;
    font-size: 14px;
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    color: white;
    box-shadow: 0 8px 20px rgba(37,99,235,0.22);
}

.stButton > button:hover {
    color: white;
}


/* ================= RESULT ================= */

.result-card {
    background: linear-gradient(145deg, #ffffff, #f8fafc);
    border: 1px solid #dbeafe;
    border-radius: 22px;
    padding: 28px;
    margin-top: 25px;
    box-shadow: 0 15px 45px rgba(37,99,235,0.08);
}

.result-label {
    color: #64748b;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.5px;
}

.result-name {
    font-size: 30px;
    font-weight: 800;
    color: #111827;
    margin-top: 6px;
}

.confidence-label {
    color: #64748b;
    font-size: 13px;
    margin-top: 18px;
}

.confidence-value {
    color: #2563eb;
    font-size: 22px;
    font-weight: 800;
    margin-top: 3px;
}


/* ================= INFO CARDS ================= */

.info-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 22px;
    height: 100%;
    box-shadow: 0 8px 25px rgba(15,23,42,0.04);
}

.info-icon {
    font-size: 26px;
    margin-bottom: 10px;
}

.info-title {
    color: #111827;
    font-size: 16px;
    font-weight: 700;
    margin-bottom: 7px;
}

.info-text {
    color: #64748b;
    font-size: 13px;
    line-height: 1.6;
}


/* ================= MODEL INFO ================= */

.metric-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 22px 10px;
    text-align: center;
    box-shadow: 0 8px 25px rgba(15,23,42,0.04);
}

.metric-value {
    color: #111827;
    font-size: 24px;
    font-weight: 800;
}

.metric-label {
    color: #64748b;
    font-size: 12px;
    margin-top: 5px;
}


/* ================= CLASS CARDS ================= */

.class-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 17px 10px;
    text-align: center;
    color: #334155;
    font-weight: 700;
    box-shadow: 0 6px 18px rgba(15,23,42,0.03);
}


/* ================= DISCLAIMER ================= */

.disclaimer {
    background: #fff7ed;
    border: 1px solid #fed7aa;
    border-radius: 16px;
    padding: 18px 20px;
    color: #9a3412;
    font-size: 12px;
    line-height: 1.7;
    margin-top: 30px;
}


/* ================= FOOTER ================= */

.footer {
    text-align: center;
    padding: 35px 10px 10px;
    color: #64748b;
    font-size: 12px;
    line-height: 1.7;
}

.footer-title {
    color: #334155;
    font-size: 14px;
    font-weight: 700;
}


/* ================= MOBILE ================= */

@media (max-width: 768px) {

    .hero-title {
        font-size: 34px;
    }

    .hero {
        padding-top: 20px;
    }

    .upload-card {
        padding: 20px;
    }

    .nav-badge {
        display: none;
    }

}

</style>
""", unsafe_allow_html=True)


# =========================================================
# NAVBAR
# =========================================================

st.html("""
<div class="navbar">

    <div class="logo">
        <div class="logo-icon">🧠</div>
        <span>NeuroScan</span>
    </div>

    <div class="nav-badge">
        CNN • Deep Learning
    </div>

</div>
""")


# =========================================================
# HERO
# =========================================================

st.html("""
<div class="hero">

    <div class="hero-tag">
        AI-POWERED MRI ANALYSIS
    </div>

    <div class="hero-title">
        Brain Tumour<br>
        <span class="hero-highlight">Classification</span>
    </div>

    <div class="hero-description">
        Upload a brain MRI image and let a trained Convolutional
        Neural Network analyze it across four classification categories.
    </div>

</div>
""")


# =========================================================
# UPLOAD SECTION
# =========================================================

st.html("""
<div class="section-title">
    Analyze Your MRI
</div>

<div class="section-subtitle">
    Upload an MRI image to get a CNN-based prediction
</div>
""")

st.html("""
<div class="upload-card">

    <div class="upload-title">
        Upload MRI Image
    </div>

    <div class="upload-description">
        Supported formats: JPG, JPEG, PNG • Maximum size: 10 MB
    </div>

</div>
""")


uploaded_file = st.file_uploader(
    "Choose MRI Image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)


# =========================================================
# IMAGE PREVIEW
# =========================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.write("")

    preview_col, details_col = st.columns(
        [1, 1],
        gap="large"
    )

    with preview_col:

        st.image(
            image,
            caption="MRI Preview",
            use_container_width=True
        )

    with details_col:

        st.html(f"""
        <div class="card">

            <div class="info-icon">📄</div>

            <div class="info-title">
                {uploaded_file.name}
            </div>

            <div class="info-text">
                File size: {uploaded_file.size / 1024:.2f} KB
                <br><br>
                Image is ready for analysis.
            </div>

        </div>
        """)

        st.write("")

        analyze_button = st.button(
            "🔍  Analyze MRI",
            type="primary"
        )

    # =====================================================
    # PREDICTION
    # =====================================================

    if analyze_button:

        with st.spinner("Analyzing MRI image..."):

            processed_image = image.resize((128, 128))

            image_array = np.array(processed_image)

            image_array = image_array / 255.0

            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            predictions = model.predict(
                image_array,
                verbose=0
            )

            probabilities = predictions[0]

            predicted_index = np.argmax(
                probabilities
            )

            predicted_class = class_names[
                predicted_index
            ]

            confidence = (
                float(
                    probabilities[predicted_index]
                ) * 100
            )

        # =================================================
        # RESULT CARD
        # =================================================

        st.html(f"""
        <div class="result-card">

            <div class="result-label">
                PREDICTION RESULT
            </div>

            <div class="result-name">
                {display_names[predicted_class]}
            </div>

            <div class="confidence-label">
                Model Confidence
            </div>

            <div class="confidence-value">
                {confidence:.2f}%
            </div>

        </div>
        """)

        st.progress(
            min(confidence / 100, 1.0)
        )

        # =================================================
        # PROBABILITIES
        # =================================================

        st.write("")

        st.html("""
        <div class="section-title">
            Class Probabilities
        </div>
        """)

        probability_cols = st.columns(4)

        for i, class_name in enumerate(class_names):

            probability = (
                float(probabilities[i]) * 100
            )

            with probability_cols[i]:

                st.html(f"""
                <div class="class-card">

                    {display_names[class_name]}

                    <br><br>

                    <span style="
                        font-size:22px;
                        color:#2563eb;
                        font-weight:800;
                    ">
                        {probability:.2f}%
                    </span>

                </div>
                """)


# =========================================================
# HOW IT WORKS
# =========================================================

st.divider()

st.html("""
<div class="section-title">
    How It Works
</div>

<div class="section-subtitle">
    From MRI upload to CNN prediction
</div>
""")


steps = [
    (
        "📤",
        "Upload",
        "Upload a brain MRI image in JPG, JPEG or PNG format."
    ),
    (
        "⚙️",
        "Preprocess",
        "The image is resized to 128×128 and normalized."
    ),
    (
        "🧠",
        "CNN Analysis",
        "The trained CNN processes the MRI image."
    ),
    (
        "📊",
        "Prediction",
        "The model returns the predicted class and probabilities."
    )
]


step_cols = st.columns(4)

for col, step in zip(step_cols, steps):

    with col:

        st.html(f"""
        <div class="info-card">

            <div class="info-icon">
                {step[0]}
            </div>

            <div class="info-title">
                {step[1]}
            </div>

            <div class="info-text">
                {step[2]}
            </div>

        </div>
        """)


# =========================================================
# MODEL INFORMATION
# =========================================================

st.divider()

st.html("""
<div class="section-title">
    Model Information
</div>

<div class="section-subtitle">
    Details about the trained CNN model
</div>
""")


model_info = [
    ("CNN", "Model"),
    ("4", "Classes"),
    ("128 × 128", "Input Size"),
    ("93.77%", "Test Accuracy")
]


model_cols = st.columns(4)

for col, (value, label) in zip(
    model_cols,
    model_info
):

    with col:

        st.html(f"""
        <div class="metric-card">

            <div class="metric-value">
                {value}
            </div>

            <div class="metric-label">
                {label}
            </div>

        </div>
        """)


# =========================================================
# SUPPORTED CLASSIFICATIONS
# =========================================================

st.divider()

st.html("""
<div class="section-title">
    Supported Classifications
</div>

<div class="section-subtitle">
    The CNN is trained to classify four categories
</div>
""")


class_cols = st.columns(4)

for col, class_name in zip(
    class_cols,
    class_names
):

    with col:

        st.html(f"""
        <div class="class-card">
            🧠 {display_names[class_name]}
        </div>
        """)


# =========================================================
# DISCLAIMER
# =========================================================

st.html("""
<div class="disclaimer">

    ⚠️ <strong>Disclaimer:</strong>
    This application is developed for educational and research
    purposes only. It is not intended for medical diagnosis,
    clinical decision-making, or replacement of professional
    medical advice.

</div>
""")


# =========================================================
# FOOTER
# =========================================================

st.html("""
<div class="footer">

    <div class="footer-title">
        🧠 NeuroScan
    </div>

    Brain Tumour Classification • CNN Deep Learning Project
    <br>

    Built with Python, Flask & TensorFlow

</div>
""")