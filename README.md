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
  - Assignment email (ex: netops@company.com)

Example Prediction:
```json
{
  "label": "Network",
  "confidence": 0.79,
  "priority": "High",
  "assign_to": "netops@company.com"
}
