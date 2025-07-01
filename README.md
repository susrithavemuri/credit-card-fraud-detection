# 💳 Credit Card Fraud Detection

A complete end-to-end machine learning pipeline to detect fraudulent credit card transactions. Built using Python, scikit-learn, and Streamlit, this project demonstrates data preprocessing, logistic regression modeling, class imbalance handling using under-sampling and ensemble techniques, and a live prediction interface through a Streamlit app.

---

## 🚀 Project Highlights

- ✅ Achieved **87% accuracy** in detecting fraud using logistic regression.
- 🧠 Reduced false positives by **16%** through feature engineering and hyperparameter tuning.
- ⚖️ Used **RandomUnderSampler** and **Bagging Classifier** to address data imbalance.
- 📈 Evaluated models using classification report and ROC AUC.
- 🌐 Deployed an interactive **Streamlit web app** to upload and analyze new transaction data.
- 🧩 Modularized code using reusable components in `src/`.

---



## ⚙️ How to Run Locally

1. **Clone the repo:*
'git clone https://github.com/susrithavemuri/credit-card-fraud-detection.git
cd credit-card-fraud-detection

2 Create a virtual environment:
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
Install dependencies:
pip install -r requirements.txt


Train the model (optional):
Run the Jupyter notebooks under notebooks/ to train and evaluate the model.
Or, use the provided trained model:
model_lr.pkl

3.Run the Streamlit app:
streamlit run app.py



## 📊 Model Details
- Algorithm: Logistic Regression
- Techniques Used: StandardScaler, Under-Sampling, BaggingClassifier
- Evaluation: ROC AUC Score, Classification Report
- Handling Imbalance: RandomUnderSampler from imblearn
- Live Prediction: via Streamlit with file uploader


## 🔍 Dataset Info
- Source: Kaggle Credit Card Fraud Dataset
- Details: 284,807 transactions made by European cardholders in 2013. Contains only 492 frauds (0.17%).





## 👩‍💻 Author
- Sai Susritha Vemuri
- GitHub: @susrithavemuri
- LinkedIn: linkedin.com/in/susritha-sai-vemuri-b2bab4269



## 🛡️ License
This project is for educational purposes. Dataset belongs to its original creators.




