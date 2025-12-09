import io
from typing import List, Optional, Tuple

import numpy as np
import streamlit as st
import torch
# Fix for specific streamlit/torch issue
torch.classes.__path__ = []
import torch.nn as nn
from PIL import Image

from pathlib import Path

# --- CONFIGURATION ---
MODEL_FILENAME = "SugarcaneLeaf_MobileNetV3Large.pth"

# Fixed class mapping for Sugarcane
# Using display-friendly names
CLASS_TO_LABEL = {
    0: "Healthy",
    1: "Mosaic",
    2: "Red Rot",
    3: "Rust",
    4: "Yellow Leaf"
}

# --- BACKEND LOGIC (UNCHANGED) ---

def _infer_num_classes_from_state(state_dict: dict) -> Optional[int]:
    candidates = [
        "classifier.3.weight", 
        "classifier.2.weight",
        "head.fc.weight",
        "fc.weight",
        "classifier.weight",
    ]
    for k in candidates:
        if k in state_dict:
            return int(state_dict[k].shape[0])
    
    # Fallback search
    keys = [k for k in state_dict.keys() if k.endswith(".weight")]
    for k in keys:
        if ".classifier" in k or ".head" in k or k.endswith("fc.weight"):
            try:
                return int(state_dict[k].shape[0])
            except Exception:
                pass
    return None

@st.cache_resource(show_spinner=True)
def load_model(weights_path: str) -> Tuple[nn.Module, List[str]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    try:
        ckpt = torch.load(weights_path, map_location=device)
    except FileNotFoundError:
        st.error(f"Model file `{weights_path}` not found. Please place it in the root directory.")
        st.stop()

    if isinstance(ckpt, dict):
        state_dict = ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt
    else:
        state_dict = ckpt

    # Determine classes
    if CLASS_TO_LABEL:
        num_classes = len(CLASS_TO_LABEL)
        class_names = [CLASS_TO_LABEL[i] for i in range(num_classes)]
    else:
        num_classes = _infer_num_classes_from_state(state_dict) or 5
        class_names = [f"Class {i}" for i in range(num_classes)]

    model = None
    errors = []

    # Initialize MobileNetV3 Large
    try:
        from torchvision.models import mobilenet_v3_large
        tv_model = mobilenet_v3_large(weights=None)
        
        in_features = tv_model.classifier[-1].in_features
        tv_model.classifier[-1] = nn.Linear(in_features, num_classes)
        
        tv_model.load_state_dict(state_dict, strict=False)
        model = tv_model
    except Exception as e:
        errors.append(f"MobileNetV3 load failed: {e}")

    if model is None:
        raise RuntimeError(
            "Failed to load model. " + " ; ".join(errors)
        )

    model.to(device)
    model.eval()
    return model, class_names

def preprocess_image(img: Image.Image) -> torch.Tensor:
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    # Standard input size for MobileNetV3 is usually 224
    img = img.resize((224, 224))

    arr = np.array(img).astype("float32") / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))
    tensor = torch.from_numpy(arr)
    return tensor

def predict(model: nn.Module, tensor: torch.Tensor) -> Tuple[int, float, np.ndarray]:
    device = next(model.parameters()).device
    with torch.no_grad():
        logits = model(tensor.unsqueeze(0).to(device))
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
    return idx, conf, probs

# --- PAGE UI & SETUP ---
st.set_page_config(page_title="LeafLens - Sugarcane Health", page_icon="🌿", layout="wide")

# Custom CSS for Professional Green Theme (Dark Mode Compatible)
st.markdown("""
<style>
    /* CSS Variables to handle both Light and Dark modes.
       This ensures text is always readable against its background.
    */
    :root {
        --primary-green: #1b5e20;        /* Dark Green for Light Mode Headers */
        --highlight-green: #43a047;      /* Standard Button/Accent Green */
        --box-bg: #e8f5e9;               /* Light Green Box Background */
        --box-text: #1b5e20;             /* Dark Text for Light Box */
        --tab-bg: #e8f5e9;
        --metric-color: #2e7d32;
    }

    /* DARK MODE OVERRIDES */
    @media (prefers-color-scheme: dark) {
        :root {
            --primary-green: #81c784;    /* Light Green for Dark Mode Headers (Readable on Black) */
            --highlight-green: #4caf50;  
            --box-bg: #1b3320;           /* Dark Green Box Background */
            --box-text: #e8f5e9;         /* Light Text for Dark Box */
            --tab-bg: #263238;
            --metric-color: #81c784;
        }
    }

    /* Global Fonts */
    h1, h2, h3 {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: var(--primary-green) !important;
    }
    
    /* Metrics Styling */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        color: var(--metric-color) !important;
    }
    
    /* Tabs Styling - forcing better contrast */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: var(--tab-bg);
        border-radius: 5px;
        padding: 10px 20px;
        color: inherit; /* Let Streamlit decide text color based on theme */
    }
    .stTabs [aria-selected="true"] {
        background-color: var(--highlight-green) !important;
        color: white !important;
    }

    /* Custom Button */
    div.stButton > button:first-child {
        background-color: var(--highlight-green);
        color: white;
        border-radius: 8px;
        padding: 0.5em 1em;
        font-weight: bold;
        border: none;
    }
    div.stButton > button:first-child:hover {
        opacity: 0.9;
    }
    
    /* Info Box with Adaptive Colors */
    .info-box {
        background-color: var(--box-bg);
        color: var(--box-text);
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid var(--highlight-green);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Resolve static asset directory
APP_DIR = Path(__file__).parent.resolve()
_public_candidates = [
    APP_DIR / "public",
    Path.cwd() / "public",
    APP_DIR.parent / "public",
]
PUBLIC_DIR = next((p for p in _public_candidates if p.exists()), _public_candidates[0])

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/sugarcane.png", width=80)
    st.title("LeafLens AI")
    st.markdown("### Model Info")
    st.info(f"**Architecture:** MobileNetV3 Large\n\n**Weights:** `{MODEL_FILENAME}`")
    
    st.markdown("---")
    st.write("LeafLens is an intelligent tool for early detection of sugarcane leaf diseases using Computer Vision.")
    
    st.link_button("📂 GitHub Repository", "https://github.com/iZekCode/LeafLens")
    
    st.markdown("### Performance Metrics")
    with st.expander("Show Charts"):
        shown_any = False
        for rel, label in [    
            ("acc.png", "Accuracy Curve"),
            ("loss.png", "Loss Curve"),
            ("cm.png", "Confusion Matrix")
        ]:
            img_path = PUBLIC_DIR / rel
            if img_path.exists():
                st.caption(label)
                st.image(str(img_path), use_column_width=True)
                shown_any = True
        
        if not shown_any:
            st.caption("No charts found in public/ folder.")

# --- HERO SECTION ---
col_hero_text, col_hero_img = st.columns([2, 1])

with col_hero_text:
    st.title("LeafLens: Protect Your Yield")
    st.markdown("""
    <div class="info-box">
    <b>Welcome!</b><br>
    Healthy crops mean a healthy harvest. With <b>LeafLens</b>, 
    farmers and researchers can analyze leaf images in seconds to identify 
    common diseases like <b>Mosaic, Red Rot, Rust, and Yellow Leaf</b>.
    </div>
    """, unsafe_allow_html=True)

with col_hero_img:
    hero_local = PUBLIC_DIR / "1.png"
    if hero_local.exists():
        st.image(str(hero_local), use_column_width=True, caption="Sugarcane Field")

# --- DISEASE INFO (TABS) ---
st.subheader("📚 Disease Encyclopedia")
st.caption("Learn about common sugarcane leaf conditions:")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["✅ Healthy", "🦠 Mosaic", "🍂 Red Rot", "🔸 Rust", "⚠️ Yellow Leaf"])

with tab1:
    st.success("**Healthy Plant**")
    st.write("The plant shows vibrant green color with no visible spots, discoloration, or drying tips.")

with tab2:
    st.error("**Mosaic Disease**")
    col_d1, col_d2 = st.columns([1, 2])
    with col_d2:
        st.write("""
        **Symptoms:** Mottled patterns of light and dark green on leaves.
        \n**Impact:** Can stunt growth and reduce sugar content significantly.
        """)

with tab3:
    st.error("**Red Rot**")
    col_d1, col_d2 = st.columns([1, 2])
    with col_d2:
        st.write("""
        **Symptoms:** Reddening of internal tissues and drying of leaves starting from the top.
        \n**Impact:** One of the most destructive diseases, often causing total stalk failure.
        """)

with tab4:
    st.error("**Rust**")
    col_d1, col_d2 = st.columns([1, 2])
    with col_d2:
        st.write("""
        **Symptoms:** Small, elongated orange or brown pustules on the leaf surface.
        \n**Impact:** Reduces photosynthesis and overall plant vigor.
        """)

with tab5:
    st.error("**Yellow Leaf**")
    col_d1, col_d2 = st.columns([1, 2])
    with col_d2:
        st.write("""
        **Symptoms:** Intense yellowing of the midrib, spreading to the leaf blade.
        \n**Impact:** Severe yield loss if not managed early.
        """)

st.divider()

# --- PREDICTION SECTION ---
st.header("🔍 Image Analysis")

# Load Model
try:
    model, class_names = load_model(MODEL_FILENAME)
except Exception as e:
    st.error("Failed to load model logic.")
    st.exception(e)
    st.stop()

col_upload, col_result = st.columns([1, 1.2])

with col_upload:
    st.markdown("#### 1. Upload Photo")
    uploaded = st.file_uploader(
        "Supported formats: PNG, JPG, JPEG", 
        type=["png", "jpg", "jpeg"], 
        accept_multiple_files=False
    )
    
    if uploaded is not None:
        image_bytes = uploaded.read()
        img = Image.open(io.BytesIO(image_bytes))
        st.image(img, caption="Uploaded Preview", use_column_width=True)

with col_result:
    st.markdown("#### 2. Diagnosis Result")
    
    if uploaded is None:
        st.info("Please upload an image in the left panel to begin analysis.")
    
    else:
        if st.button("Start Analysis 🚀", type="primary", use_container_width=True):
            with st.spinner("Analyzing leaf patterns..."):
                tensor = preprocess_image(img)
                idx, conf, probs = predict(model, tensor)

            pred_label = class_names[idx] if idx < len(class_names) else f"Class {idx}"

            # Result Container
            st.markdown("---")
            
            # Determine logic for display
            is_healthy = "Healthy" in pred_label
            
            if is_healthy:
                st.success(f"### Result: {pred_label}")
                st.balloons()
            else:
                st.error(f"### Detected: {pred_label}")
            
            st.write(f"**Confidence Score:** {conf:.1%}")
            st.progress(conf)
            
            # Show breakdown details
            with st.expander("View Probability Breakdown"):
                for label, prob in zip(class_names, probs):
                    col_p1, col_p2 = st.columns([3, 1])
                    with col_p1:
                        st.write(label)
                        st.progress(float(prob))
                    with col_p2:
                        st.write(f"{prob:.1%}")

            # Low confidence warning
            if conf < 0.6:
                st.warning("⚠️ **Note:** Model confidence is below 60%. The result might be inaccurate. Ensure the image has good lighting and focuses clearly on the leaf.")