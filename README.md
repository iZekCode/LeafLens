# Chest CT Scan Classifier (Streamlit)

A simple, good-looking Streamlit web app that loads a MobileNetV3-Large PyTorch model from `SugarcaneLeaf_MobileNetV3Large.pth` and predicts the class probabilities for an uploaded sugarcane leaf image.

## Features
- Upload PNG/JPG sugarcane leaf images
- Auto-preprocessing for MobileNetV3Large (resize 224, ImageNet normalization)
- Top-K predictions with confidence bar chart
- Generic labels generated automatically (Class 0..N)

# (Optional) Create & activate virtual environment
python -m venv .venv; .\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Model
- Place the trained checkpoint at `SugarcaneLeaf_MobileNetV3Large.pth` (already present).
- The app tries to infer the number of classes from the checkpoint's classifier weight shape.

## Notes
- This app is not a medical device; predictions are probabilistic.
- If your model uses a different preprocessing pipeline, adjust the `PREPROCESS` transform in `app.py`.
