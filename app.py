import io
from typing import List, Optional, Tuple

import numpy as np
import streamlit as st
import torch
torch.classes.__path__ = []
import torch.nn as nn
from PIL import Image

from pathlib import Path

# --- CONFIGURATION ---
MODEL_FILENAME = "SugarcaneLeaf_MobileNetV3Large.pth"

# Fixed class mapping for Sugarcane
CLASS_TO_LABEL = {
    0: "Healthy",
    1: "Mosaic",
    2: "RedRot",
    3: "Rust",
    4: "Yellow"
}

def _infer_num_classes_from_state(state_dict: dict) -> Optional[int]:
    candidates = [
        "classifier.3.weight", # MobileNetV3 Large often uses this index
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
        
        # Adjust classifier head for correct number of classes
        # MobileNetV3 classifier structure: Sequential(Linear, Hardswish, Dropout, Linear)
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

# --- PAGE SETUP ---
st.set_page_config(page_title="LeafLens - Sugarcane Health", page_icon="🌿", layout="centered")

# Custom CSS for Green/Nature Theme
st.markdown("""
<style>
    /* Main Background & Fonts */
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
    }
    h1 {
        font-weight: 700;
        color: #2E7D32; 
    }
    
    /* Custom Button for "Start Detecting" */
    a.custom-btn {
        display: inline-block;
        padding: 0.6em 1.2em;
        margin-top: 20px;
        color: #ffffff !important;
        background-color: #43a047; /* Leaf Green */
        border-radius: 8px;
        text-decoration: none;
        font-weight: 600;
        text-align: center;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    a.custom-btn:hover {
        background-color: #2e7d32;
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
        color: #1b5e20;
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

# --- HERO SECTION ---
st.title("LeafLens: Protect Your Yield")
st.caption("Intelligent Sugarcane Disease Detection")

col1, col2 = st.columns([2, 1])
with col1:
    st.markdown(
        """
        <div style="font-size: 1.1em; color: #444; line-height: 1.6;">
        Healthy crops mean a healthy harvest. With <b>LeafLens</b>, 
        farmers and researchers can analyze leaf images in seconds to identify 
        common diseases like <b>Mosaic, Red Rot, and Rust</b>.
        <br><br>
        Upload a leaf photo below to get an instant diagnosis.
        </div>
        """, unsafe_allow_html=True
    )
    st.markdown('<a href="#prediction-section" class="custom-btn">Start Diagnosis</a>', unsafe_allow_html=True)

with col2:
    # Ensure you have a relevant image named 1.png in public/ folder (e.g. a sugarcane field)
    hero_local = PUBLIC_DIR / "1.png" 
    if hero_local.exists():
        st.image(str(hero_local), use_column_width=True)
    else:
        st.info("Add '1.png' to public folder")

# --- INFO SECTION ---
st.divider()
st.header("Identify Common Sugarcane Diseases")

st.write(
    "Early detection of fungal and viral diseases can prevent spread and save your harvest. "
    "<b>LeafLens</b> is trained to recognize the following conditions:"
)

# Row 1: Mosaic & RedRot
row1_left, row1_right = st.columns(2)
with row1_left:
    with st.container(border=True):
        st.subheader("Mosaic")
        st.write(
            "**Symptoms:** Mottled patterns of light and dark green on leaves.\n\n"
            "**Impact:** Can stunt growth and reduce sugar content."
        )
with row1_right:
    with st.container(border=True):
        st.subheader("Red Rot")
        st.write(
            "**Symptoms:** Reddening of internal tissues and drying of leaves.\n\n"
            "**Impact:** One of the most destructive diseases, often causing total stalk failure."
        )

# Row 2: Rust & Yellow
row2_left, row2_right = st.columns(2)
with row2_left:
    with st.container(border=True):
        st.subheader("Rust")
        st.write(
            "**Symptoms:** Small, elongated orange or brown pustules on the leaf surface.\n\n"
            "**Impact:** Reduces photosynthesis and overall plant vigor."
        )
with row2_right:
    with st.container(border=True):
        st.subheader("Yellow Leaf")
        st.write(
            "**Symptoms:** Intense yellowing of the midrib, spreading to the leaf blade.\n\n"
            "**Impact:** severe yield loss if not managed early."
        )

# Row 3: Healthy
with st.container(border=True):
    st.subheader("Healthy")
    st.write(
        "The plant shows vibrant green color with no visible spots, discoloration, or drying tips."
    )

st.divider()

# --- PREDICTION SECTION ---
st.markdown('<div id="prediction-section"></div>', unsafe_allow_html=True)

st.title("LeafLens Classifier")

# Sidebar for Model Info
with st.sidebar:
    st.subheader("LeafLens AI")
    st.write(f"Weights: `{MODEL_FILENAME}`")
    st.write("Architecture: **MobileNetV3 Large**")
 
    st.link_button("GitHub Repository", "https://github.com/Jasonnn13/FinalProjectComputerVision")

    st.subheader("Performance Metrics")
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
        st.caption("Add 'acc.png' and 'loss.png' to the public/ folder to see charts here.")

# Load Model
try:
    model, class_names = load_model(MODEL_FILENAME)
except Exception as e:
    st.error("Failed to load model logic.")
    st.exception(e)
    st.stop()

uploaded = st.file_uploader(
    "Upload Sugarcane Leaf Image (PNG/JPG)", type=["png", "jpg", "jpeg"], accept_multiple_files=False
)

if uploaded is not None:
    image_bytes = uploaded.read()
    img = Image.open(io.BytesIO(image_bytes))
    
    # Center the image display
    col_img, _ = st.columns([1, 2]) 
    with col_img:
        st.image(img, caption="Uploaded Sample", use_column_width=True)

    if st.button("Analyze with LeafLens", type="primary"):
        with st.spinner("LeafLens is analyzing patterns..."):
            tensor = preprocess_image(img)
            idx, conf, probs = predict(model, tensor)

        pred_label = class_names[idx] if idx < len(class_names) else f"Class {idx}"

        st.markdown("---")
        st.subheader("Diagnosis Result")
        
        # dynamic color based on result
        res_color = "green" if pred_label == "Healthy" else "red"
        
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.markdown(f":{res_color}[**{pred_label}**]")
        with col_res2:
            st.metric("Confidence Score", f"{conf:.2%}")
            
        # Optional: Show breakdown if confidence is low
        if conf < 0.7:
            st.warning("Confidence is low. Check the breakdown below:")
            st.bar_chart({label: prob for label, prob in zip(class_names, probs)})

else:
    st.info("Please upload an image to begin diagnosis.")