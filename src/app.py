# src/app.py
from fastapi import FastAPI, Query
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import os, time, pathlib, joblib, numpy as np, requests

# -------- CONFIG --------
MODEL_PATH = pathlib.Path("models/classifier.joblib")
WEBHOOK = os.getenv("WEBHOOK_URL")                      # Slack webhook (optional)
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.65"))

TEAM_MAP = {
    "Network":  "netops@company.com",
    "Access":   "iam@company.com",
    "Software": "apps@company.com",
    "Hardware": "field-it@company.com",
    "Billing":  "fin-ops@company.com",
}

HIGH_PRIORITY_KEYWORDS = [
    "urgent","immediately","down","cannot login","can't login",
    "payment charged twice","production","server down","site down"
]

app = FastAPI(title="IT Ticket Classifier")
model = None

# -------- MODELS --------
class Ticket(BaseModel):
    message: str

def is_high_priority(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in HIGH_PRIORITY_KEYWORDS)

def notify_slack(label: str, text: str, conf: float, assign_to: str, priority: str):
    if not WEBHOOK:
        return
    payload = {
        "text": (
            f":ticket: *Ticket classified*\n"
            f"• *Label:* {label}\n"
            f"• *Confidence:* {conf:.2f}\n"
            f"• *Priority:* {priority}\n"
            f"• *Assign to:* {assign_to}\n"
            f"• *Text:* {text[:300]}"
        )
    }
    try:
        r = requests.post(WEBHOOK, json=payload, timeout=5)
        print("[SLACK] status:", r.status_code)
    except Exception as e:
        print("[SLACK ERROR]", e)

# -------- STARTUP --------
@app.on_event("startup")
def load_model():
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError("Model not found. Run: python src/train.py")
    model = joblib.load(MODEL_PATH)
    print("Model loaded ✅ | Webhook configured:", bool(WEBHOOK))

# -------- ROUTES --------
@app.get("/")
def root():
    return {"status": "ok", "docs": "/docs"}

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/notify-test")
def notify_test(text: str = Query("Hello from API")):
    notify_slack("Test", text, 1.0, "it-helpdesk@company.com", "Test")
    return {"sent": True, "to": "slack", "text": text}

@app.post("/classify")
def classify(t: Ticket):
    start = time.time()
    pred_label = model.predict([t.message])[0]

    if hasattr(model.named_steps.get("clf"), "predict_proba"):
        conf = float(np.max(model.predict_proba([t.message])[0]))
    else:
        conf = 1.0

    low_conf = conf < CONF_THRESHOLD
    priority = "High" if is_high_priority(t.message) else ("Review" if low_conf else "Normal")
    assign_to = "it-helpdesk@company.com" if low_conf else TEAM_MAP.get(pred_label, "it-helpdesk@company.com")

    notify_slack(pred_label, t.message, conf, assign_to, priority)

    return JSONResponse({
        "label": pred_label,
        "confidence": round(conf, 4),
        "priority": priority,
        "assign_to": assign_to,
        "latency_ms": int((time.time() - start) * 1000),
    })
