# src/app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pathlib
import numpy as np

MODEL_PATH = pathlib.Path("models/classifier.joblib")

app = FastAPI(title="IT Ticket Classifier - Simple ML")

model = None

class Ticket(BaseModel):
    message: str

class Output(BaseModel):
    label: str
    confidence: float

@app.on_event("startup")
def load_model():
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError("Model not found. Run: python src/train.py first")
    model = joblib.load(MODEL_PATH)
    print("model loaded")

@app.get("/")
def root():
    return {"msg": "API is running"}

@app.post("/classify", response_model=Output)
def classify(ticket: Ticket):
    pred = model.predict([ticket.message])[0]
    if hasattr(model.named_steps["clf"], "predict_proba"):
        proba = model.predict_proba([ticket.message])[0]
        conf = float(np.max(proba))
    else:
        conf = 1.0
    return {"label": pred, "confidence": round(conf, 4)}
