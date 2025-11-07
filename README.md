Support Ticket Classification using AI (NLP + FastAPI + DevOps)

This project automatically classifies customer support tickets using Machine Learning.
It predicts the ticket category, priority and assigns it to the right support team.

🔥 Problem We Solve

Support teams get thousands of tickets daily.
Manually reading and classifying each ticket takes time.

Our system does this automatically.

🤖 AI / ML Implementation

Preprocessed ticket text using NLP (stopwords removal, tokenization)

Converted text to features using TF-IDF

Trained machine learning model for classification

Model returns:

Category (Network / Payment / Login / etc.)

Confidence Score

Priority (Low / Medium / High)

Recommended Assignment team email

🛠 Tech Stack
Section	Tools Used
Programming	Python
ML Libraries	scikit-learn, pandas, numpy
Backend API	FastAPI
Testing	Swagger UI / Postman
DevOps	FastAPI deployment, Health checks, Dockerfile
⚙️ DevOps Implementation

We deployed the ML model as a FastAPI service so that any application can call the API.

DevOps work we did:

Created API endpoint /classify to serve model predictions

Created health endpoint /health to check service status

Created Dockerfile (containerization ready)

Can run on cloud server / Render / AWS EC2

Can be integrated into Slack / Helpdesk systems

So DevOps ensures the model is not only trained but also runs continuously as a reliable service.

🚀 How to Run Locally
Install dependencies
pip install -r requirements.txt

Start server
uvicorn app:app --reload --port 8000

Open API Docs

http://localhost:8000/docs

🧪 Example API Request
{
  "text": "Production website is not loading for users"
}

Output Response
{
  "label": "Network",
  "confidence": 0.79,
  "priority": "High",
  "assign_to": "netops@company.com"
}


🏁 Conclusion

This project reduces manual work for support teams, speeds up ticket classification, and smartly routes issues to the right team instantly.
