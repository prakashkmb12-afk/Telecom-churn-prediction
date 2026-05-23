# 📞 Telecom Customer Churn Prediction System

An end-to-end industry-grade machine learning project designed to proactively identify telecom customers at a high risk of churning. By predicting which customers are likely to discontinue service, telecom companies can design targeted retention strategies, special discounts, and personalized offers to reduce revenue loss.

---

## 🚀 Key Features

- **End-to-End ML Pipeline**: Seamlessly runs data cleaning, train-test splitting, feature encoding, scaling, imbalance resolution, baseline comparison, and hyperparameter tuning.
- **Advanced Class Imbalance Handling**: Utilizes **SMOTEENN** (SMOTE + Edited Nearest Neighbors) to balance churn classes for higher sensitivity.
- **Multi-Model Comparison**: Evaluates six high-performance classifiers:
  1. *Logistic Regression*
  2. *Decision Tree*
  3. *Random Forest*
  4. *XGBoost*
  5. *LightGBM*
  6. *CatBoost*
- **Hyperparameter Optimization**: Uses **Optuna** to perform Bayesian optimization on the top-performing algorithm.
- **Interactive Streamlit Web App**:
  - **Single Customer Predictor**: Input profile elements in real-time, compute a churn probability gauge, and get customized retention recommendations.
  - **Batch Predictor (CSV Upload)**: Upload a CSV containing multiple records, get bulk churn rates, and download predicted results.
  - **Model Insights Dashboard**: View confusion matrix, ROC curves, baseline comparisons, and feature importances.
  - **EDA Tab**: Dynamic statistical data exploration, tenure distributions, and correlation maps.

---

## 📂 Project Structure

```text
telecom/
├── app/
│   └── app.py                  # Streamlit Web Application
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Raw Dataset (Downloaded)
├── models/                     # Saved Models & Diagnostic Visualizations
│   ├── best_churn_model.joblib # Serialized fitted preprocessor & optimized model
│   ├── metrics.json            # Model performance comparison & final metrics
│   ├── confusion_matrix.png    # Evaluation Confusion Matrix
│   ├── roc_curve.png           # Evaluation ROC Curve
│   └── feature_importance.png  # Top drivers of churn chart
├── src/
│   ├── download_data.py        # Dataset Downloader
│   ├── data_preprocessing.py   # Data Cleaning & Column Transformer Pipeline
│   └── train.py                # Pipeline Comparison, Optuna Tuning, & Evaluation
├── requirements.txt            # Project Dependencies
└── README.md                   # Project Documentation
```

---

## 🛠️ Installation & Setup

### 1. Clone & Navigate to Workspace
```powershell
cd d:\Projects\telecom
```

### 2. Activate Virtual Environment
```powershell
# Create venv if not already created
python -m venv venv

# Activate on Windows
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## 🏃 Running the Pipeline

### Step 1: Download the Dataset
The dataset will be downloaded programmatically from the raw IBM database:
```powershell
python src/download_data.py
```

### Step 2: Run the Training Pipeline
Trains all six baseline classifiers, applies SMOTEENN, optimizes the best model using Optuna, and saves the final joblib pipeline along with evaluation plots:
```powershell
python src/train.py
```

### Step 3: Launch the Streamlit Web Application
Start the interactive real-time dashboard:
```powershell
streamlit run app/app.py
```

---

## 📊 Expected Output & Business Advantages

- **High Precision & Sensitivity**: Balances the tradeoff between false positives (cost of unnecessary discounts) and false negatives (cost of losing customers).
- **Early Churn Identification**: Identifies the key customer behaviors leading to churn, such as month-to-month contracts, electronic checks, or fiber optic without tech support.
- **Actionable Retention Recommendations**: Proposes customized incentives (automatic billing setup, contract lock-ins, free support months) based on each customer's specific risk factor.
