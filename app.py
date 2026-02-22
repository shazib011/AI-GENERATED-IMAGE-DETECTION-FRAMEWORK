import streamlit as st
import requests
from PIL import Image
import base64, io
import pandas as pd
from pathlib import Path

API_URL = "http://localhost:8000/predict"
LOG_PATH = Path("logs/predictions.csv")

st.set_page_config(page_title="Deepfake Detection Dashboard", layout="wide")
st.title("Deepfake Detection Framework (Face-swap + GAN)")

tab1, tab2, tab3, tab4 = st.tabs(["Upload & Detect", "History", "Metrics", "Robustness"])

with tab1:
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded Image", width=400)

        files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
        with st.spinner("Detecting..."):
            r = requests.post(API_URL, files=files, timeout=60)
        res = r.json()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Label", res["label"])
        c2.metric("Confidence", f'{res["confidence"]*100:.2f}%')
        c3.metric("Fake Prob", f'{res["prob_fake"]*100:.2f}%')
        c4.metric("Runtime", f'{res["runtime_ms"]} ms')

        st.write("Face Found:", res["face_found"])
        st.subheader("Model Probabilities")
        st.json(res["model_probs"])

        if res.get("heatmap"):
            heatmap = base64.b64decode(res["heatmap"])
            heatmap_img = Image.open(io.BytesIO(heatmap)).convert("RGB")
            st.subheader("Grad-CAM Heatmap")
            st.image(heatmap_img, width=400)

with tab2:
    st.subheader("Prediction History")
    if LOG_PATH.exists():
        df = pd.read_csv(LOG_PATH)
        st.dataframe(df.sort_values("time", ascending=False), use_container_width=True)
    else:
        st.info("No logs yet. Run a few predictions first.")

with tab3:
    st.subheader("Saved Metrics Plots (from training)")
    cm = Path("outputs/confusion_matrix.png")
    roc = Path("outputs/roc_curve.png")
    if cm.exists():
        st.image(str(cm), caption="Confusion Matrix", use_container_width=True)
    else:
        st.info("Run: python -m src.training.evaluate")
    if roc.exists():
        st.image(str(roc), caption="ROC Curve", use_container_width=True)

with tab4:
    st.subheader("Robustness Report")
    rb = Path("outputs/robustness.png")
    if rb.exists():
        st.image(str(rb), caption="Robustness Accuracy", use_container_width=True)
    else:
        st.info("Run: python -m src.training.robustness_eval")
