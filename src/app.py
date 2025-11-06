# src/app.py
from fastapi import FastAPI, Query
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import os
import time
import pathlib
import joblib
import numpy as np
import requests

# -------- Config --------
MODEL_PATH = pathlib.Path("models/classifier.joblib")
WEBHOOK = os.getenv("WEBHOOK_URL")                     # Slack/Teams webhook (optional)
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.65"))

# Team routing (edit as you like)
TEAM_MAP = {
    "Network":  "netops@company.com",
    "Access":   "iam@company.com",
    "Software": "apps@company.com",
    "Hardware": "field-it@company.com",
    "Billing":  "fin-ops@company.com",
}

# Simple keyword-based high-priority heuristic (for demo)
HIGH_PRIORITY_KEYWORDS = [
    "urgent", "immediately", "down", "cannot login", "can't login",
    "payment charged twice", "production", "server down", "site down"
]

app = FastAPI(title="IT Ticket Classifier")

model = None  # will be loaded on startup


# -------- Helpers --------
class Ticket(BaseModel):
    message: str

def is_high_priority(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in HIGH_PRIORITY_KEYWORDS)

def notify_slack(label: str, text: str, conf: float, assign_to: str, priority: str):
    """Post a simple message to Slack (if WEBHOOK is set)."""
    if not WEBHOOK:
        return
    payload = {
        "text": (
            ":ticket: *Ticket classified*\n"
            f"• *Label:* {label}\n"
            f"• *Confidence:* {conf:.2f}\n"
            f"• *Priority:* {priority}\n"
            f"• *Assign to:* {assign_to}\n"
            f"• *Text:* {text[:300]}"
        )
    }
    try:
        r = requests.post(WEBHOOK, json=payload, timeout=5)
        # Optional log line for troubleshooting:
        print(f"[SLACK] status={r.status_code}")
    except Exception as e:
        print(f"[SLACK] ERROR: {e}")


# -------- Lifecycle --------
@app.on_event("startup")
def load_model():
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError("Model not found. Run: python src/train.py")
    model = joblib.load(MODEL_PATH)
    print("Model loaded. Webhook configured:", bool(WEBHOOK))


# -------- Basic routes --------
@app.get("/")
def root():
    return {"status": "ok", "docs": "/docs"}

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/notify-test")
def notify_test(text: str = Query("Hello from API")):
    """Quick way to verify Slack integration without ML."""
    notify_slack("Test", text, 1.0, "it-helpdesk@company.com", "Test")
    return {"sent": True, "to": "slack"}


# -------- ML route --------
@app.post("/classify")
def classify(t: Ticket):
    start = time.time()

    # Predict label
    pred_label = model.predict([t.message])[0]

    # Compute confidence if the classifier supports probabilities
    if hasattr(model.named_steps.get("clf", None), "predict_proba"):
        proba = model.predict_proba([t.message])[0]
        conf = float(np.max(proba))
    else:
        conf = 1.0  # for models without proba (e.g., LinearSVC)

    # Priority + assignment
    low_conf = conf < CONF_THRESHOLD
    priority = "High" if is_high_priority(t.message) else ("Review" if low_conf else "Normal")
    assign_to = "it-helpdesk@company.com" if low_conf else TEAM_MAP.get(pred_label, "it-helpdesk@company.com")

    # Notify Slack (optional)
    notify_slack(pred_label, t.message, conf, assign_to, priority)

    return JSONResponse({
        "label": pred_label,
        "confidence": round(conf, 4),
        "priority": priority,
        "assign_to": assign_to,
        "latency_ms": int((time.time() - start) * 1000),
    })
@app.get("/notify-test")
def notify_test(text: str = "hello from api"):
    notify_slack("TEST", text, 1.0, "test@company.com", "Test")
    return {"status": "sent", "text": text}
