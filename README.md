# Support Ticket Classification using AI (NLP + FastAPI + DevOps)

This project automatically classifies support tickets into categories like **Network**, **Authentication**, **Payment**, etc.  
It predicts priority (Low/Medium/High) and which team the ticket should be assigned to.

---

## 🔥 Problem

Support teams get many tickets daily. Manually reading and tagging tickets takes time.

Our AI model does this instantly and automatically.

---

## 🤖 AI / Machine Learning (What we did)

- Cleaned ticket text (NLP preprocessing)
- Converted text to features using **TF-IDF**
- Trained machine learning classifier (Logistic Regression / SVM)
- Model outputs:
  - Category label
  - Confidence score
  - Priority level

Example Prediction:
```json
{
  "label": "Network",
  "confidence": 0.79,
  "priority": "High",
  "assign_to": "netops@company.com"
}



## 🛠 Tech Stack Used

| Area       | Tools                                   |
|------------|------------------------------------------|
| Language   | Python                                   |
| ML / NLP   | scikit-learn, pandas, numpy              |
| Backend API| FastAPI                                  |
| Testing    | Swagger UI (`/docs`), Postman            |
| DevOps     | API Deployment, Dockerfile, `/health` endpoint |

---

## ⚙️ DevOps Implementation

DevOps part is making our AI model run as a **service**.

**We implemented:**
- FastAPI `/classify` endpoint → prediction API
- `/health` endpoint → to monitor if service is running
- Dockerfile → container ready environment
- Can deploy on cloud (Render / Railway / AWS EC2 / Azure)

**DevOps ensures the model is not just trained, but accessible and running continuously.**

---

## 🚀 How to Run Locally

**Install requirements**
```bash
pip install -r requirements.txt

uvicorn app:app --reload --port 8000

http://localhost:8000/docs

