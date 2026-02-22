# Deepfake Detection Dashboard (Face-swap + GAN)

This project provides a real-time deepfake detection system for **face-swaps** and **GAN-generated** images:
- FastAPI backend for inference
- Streamlit dashboard for upload + results
- Ensemble of pretrained models + optional CNN-LSTM (patch-sequence) model
- Optional Grad-CAM explainability
- Robustness evaluation under common post-processing attacks

## 1) Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Data layout

Put your images here (or generate from datasets):
```
data/processed/
  train/real, train/fake
  val/real,   val/fake
  test/real,  test/fake
```

Labels:
- real = 0
- fake = 1 (includes face-swap and GAN)

## 3) Training (train 3 models)

Edit the bottom of `src/training/train.py` to run each model and save to:
- `weights/model_a_effnet.pth`
- `weights/model_b_resnet.pth`
- `weights/model_c_cnnlstm.pth`

Run:
```bash
python -m src.training.train
```

## 4) Evaluation plots
```bash
python -m src.training.evaluate
```
Outputs:
- `outputs/confusion_matrix.png`
- `outputs/roc_curve.png`

## 5) Robustness evaluation
```bash
python -m src.training.robustness_eval
```
Output:
- `outputs/robustness.png`

## 6) Run API + Dashboard

Terminal 1:
```bash
uvicorn api.main:app --reload --port 8000
```

Terminal 2:
```bash
streamlit run dashboard/app.py
```

## Notes
- Face detection uses OpenCV Haar Cascade for offline speed; you can replace with RetinaFace/MTCNN for better quality.
- Ensemble runs 3 models (if weights exist). If weights are missing, models will run with ImageNet heads (not meaningful) but pipeline will still work.
