# 🛡️ Fake Job Posting Detection

Detects fraudulent job postings using machine learning and visualizes insights via a Streamlit dashboard.

---

## ⚙️ Key Features & Technologies Used

### 🧠 Machine Learning
- **Model:** Logistic Regression (or Random Forest)
- **Features:** Text (title, description, etc.), categorical & boolean flags
- **Metrics:** F1-Score (important due to data imbalance)
- **Explainability:** SHAP values to interpret predictions

### 🛠️ Tools & Libraries
- **Python**
- **Pandas, NumPy, Scikit-learn**
- **SHAP for explainability**
- **Streamlit** for dashboard UI
- **Matplotlib / Seaborn** for plotting

### 📊 Dashboard Includes:
- File upload interface
- Table of predictions with probabilities
- Histogram of fraud probabilities
- Pie chart: Real vs Fake jobs
- Top-10 most suspicious listings

  ###**Future Updates**
  - Functionality to view SHAP explanation
  - Display of SHAP summary Image
 
  ## 🚀 Setup Instructions (Step-by-Step)

> 🔁 Assumes Python 3.8+ installed

### 📁 1. Clone the Repository
### 📁 2. pip install -r requirements.txt
### 📁 3. streamlit run app.py 
  

## 📁 Project Structure
fraud_detection_job/
│
├── app.py # Streamlit dashboard
├── model/
│ ├── train_model.py # Model training script
│ └── fraud_detector_pipeline.pkl # Saved model
├── requirements.txt
├── README.md
└── data/
├── train.csv
└── test.csv

**LINKS**
- **DEPLOYED LINK** - https://frauddetectionjobgit-3wnxugzqwsta8oqcpt7shd.streamlit.app/

