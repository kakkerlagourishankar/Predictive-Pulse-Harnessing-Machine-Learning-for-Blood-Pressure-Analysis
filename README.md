# 🩺 Predictive Pulse: Harnessing Machine Learning for Blood Pressure Analysis <br/>
A Flask-based web application that predicts a person's blood pressure stage (e.g., Normal, Hypertension Stage 1/2, Crisis) based on basic health inputs, using a trained ML model. The app also allows users to download the results as a PDF report.

# 🔍 Project Overview
Predictive Pulse is a machine learning project that helps estimate a user's blood pressure category from clinical features like age group, gender, systolic & diastolic BP, and other symptoms.

# Key Features:

Interactive web UI using Flask + HTML/CSS (Bootstrap)
Trained on logistic regression ML model for BP classification
Client-side PDF generation using jsPDF
Clean and responsive design
Lightweight and easy to deploy
# 🧠 How It Works
User visits the homepage and enters basic medical inputs through a form.
Model predicts the blood pressure stage (e.g., Normal, Crisis).
Advice is shown based on the prediction.
Download button allows saving the result as a PDF report.
# 🚀 Tech Stack
Frontend: HTML, CSS (Bootstrap), JavaScript (jsPDF)
Backend: Python, Flask
ML Model: Scikit-learn (logistic regression pipeline)
Data Processing: Pandas, NumPy
PDF Generator: jsPDF (client-side)
er Navigate to: http://127.0.0.1:5000
