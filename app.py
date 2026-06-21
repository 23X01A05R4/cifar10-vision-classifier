import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import os

# ─────────────────────────────────────────────
#  Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
#  Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

[data-testid="stHeader"] { background: transparent; }

.hero-title {
    text-align: center;
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.25rem;
}

.hero-subtitle {
    text-align: center;
    color: #94a3b8;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}

.glass-card {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 1.5rem;
    backdrop-filter: blur(10px);
    margin-bottom: 1rem;
}

.pred-badge {
    display: inline-block;
    background: linear-gradient(135deg, #a78bfa, #60a5fa);
    color: white;
    font-size: 1.4rem;
    font-weight: 700;
    padding: 0.4rem 1.2rem;
    border-radius: 999px;
    text-transform: capitalize;
    margin-bottom: 0.5rem;
}

.confidence-label {
    color: #94a3b8;
    font-size: 0.9rem;
}

.confidence-value {
    color: #34d399;
    font-size: 2rem;
    font-weight: 700;
}

.class-pill {
    display: inline-block;
    background: rgba(167, 139, 250, 0.15);
    border: 1px solid rgba(167, 139, 250, 0.3);
    color: #c4b5fd;
    font-size: 0.8rem;
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
    margin: 0.2rem;
}

.stFileUploader label { color: #94a3b8 !important; }
.stFileUploader [data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,0.03) !important;
    border: 2px dashed rgba(167, 139, 250, 0.4) !important;
    border-radius: 12px !important;
}

.divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.08);
    margin: 1.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  CIFAR-10 Classes & Emoji Map
# ─────────────────────────────────────────────
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

CLASS_EMOJI = {
    'airplane':    '✈️',
    'automobile':  '🚗',
    'bird':        '🐦',
    'cat':         '🐱',
    'deer':        '🦌',
    'dog':         '🐶',
    'frog':        '🐸',
    'horse':       '🐴',
    'ship':        '🚢',
    'truck':       '🚛',
}

MODEL_PATH = 'cifar10_mobilenetv2.keras'

# ─────────────────────────────────────────────
#  Model Loading (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

# ─────────────────────────────────────────────
#  Header
# ─────────────────────────────────────────────
st.markdown('<h1 class="hero-title">🔍 CIFAR-10 Image Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Powered by MobileNetV2 · Transfer Learning · TensorFlow</p>', unsafe_allow_html=True)

# Supported classes row
st.markdown(
    "<div style='text-align:center; margin-bottom:2rem;'>" +
    "".join([f'<span class="class-pill">{CLASS_EMOJI[c]} {c}</span>' for c in CLASS_NAMES]) +
    "</div>",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
#  Layout
# ─────────────────────────────────────────────
col_upload, col_results = st.columns([1, 1.6], gap="large")

with col_upload:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### 📂 Upload an Image")
    st.markdown('<p style="color:#64748b; font-size:0.85rem;">Supports JPG, PNG, WEBP</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        label="",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )

    # Also allow using the sample image
    use_sample = st.button("🖼️ Use sample image (OIP.jpg)", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Show image preview
    img_to_process = None
    image_source_name = ""

    if uploaded_file is not None:
        img_to_process = Image.open(uploaded_file).convert("RGB")
        image_source_name = uploaded_file.name
    elif use_sample and os.path.exists("OIP.jpg"):
        img_to_process = Image.open("OIP.jpg").convert("RGB")
        image_source_name = "OIP.jpg"
    elif use_sample:
        st.error("OIP.jpg not found in current directory.")

    if img_to_process is not None:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.image(img_to_process, caption=f"📷 {image_source_name}", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Prediction
# ─────────────────────────────────────────────
with col_results:
    if img_to_process is not None:
        # Check model exists
        if not os.path.exists(MODEL_PATH):
            st.error(f"❌ Model file `{MODEL_PATH}` not found. Make sure it's in the same directory as app.py.")
        else:
            with st.spinner("🧠 Running inference..."):
                model = load_model()

                # Preprocess
                img_resized = img_to_process.resize((32, 32))
                img_array = np.array(img_resized, dtype=np.float32)
                img_array = np.expand_dims(img_array, axis=0)

                predictions = model.predict(img_array, verbose=0)[0]
                top_indices = np.argsort(predictions)[::-1]

            top1_idx = top_indices[0]
            top1_class = CLASS_NAMES[top1_idx]
            top1_conf = predictions[top1_idx] * 100

            # ── Top Prediction ──
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 🏆 Top Prediction")
            st.markdown(
                f'<div class="pred-badge">{CLASS_EMOJI[top1_class]} {top1_class}</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<p class="confidence-label">Confidence</p>'
                f'<p class="confidence-value">{top1_conf:.1f}%</p>',
                unsafe_allow_html=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

            # ── All Predictions Bar Chart ──
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 📊 All Class Probabilities")

            sorted_classes = [CLASS_NAMES[i] for i in top_indices]
            sorted_confs   = [predictions[i] * 100 for i in top_indices]
            colors = [
                "#a78bfa" if i == 0 else "#60a5fa" if i == 1 else "#34d399" if i == 2 else "#475569"
                for i in range(len(sorted_classes))
            ]

            fig = go.Figure(go.Bar(
                x=sorted_confs,
                y=[f"{CLASS_EMOJI[c]} {c}" for c in sorted_classes],
                orientation='h',
                marker=dict(color=colors, line=dict(width=0)),
                text=[f"{v:.1f}%" for v in sorted_confs],
                textposition='outside',
                textfont=dict(color='white', size=12),
            ))
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white', family='Inter'),
                xaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.05)',
                    range=[0, max(sorted_confs) * 1.2],
                    ticksuffix='%',
                    tickfont=dict(color='#64748b'),
                ),
                yaxis=dict(
                    showgrid=False,
                    tickfont=dict(color='#e2e8f0', size=13),
                    autorange='reversed',
                ),
                margin=dict(l=10, r=60, t=10, b=10),
                height=360,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    else:
        # Placeholder state
        st.markdown('<div class="glass-card" style="min-height:400px; display:flex; flex-direction:column; align-items:center; justify-content:center;">', unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align:center; color:#475569; padding: 2rem;">
            <div style="font-size:4rem; margin-bottom:1rem;">🤖</div>
            <h3 style="color:#64748b; font-weight:500;">Ready to Classify</h3>
            <p style="font-size:0.9rem;">Upload an image or use the sample to see the AI in action.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Footer
# ─────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:center; color:#334155; font-size:0.8rem;">'
    'CIFAR-10 · MobileNetV2 · Transfer Learning · TensorFlow · Streamlit'
    '</p>',
    unsafe_allow_html=True
)
