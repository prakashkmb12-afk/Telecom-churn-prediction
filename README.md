# 📞 Telecom Customer Churn Prediction System
An end-to-end industry-grade machine learning project designed to proactively identify telecom customers at a high risk of churning. By predicting which customers are likely to discontinue service, telecom companies can design targeted retention strategies, special discounts, and automated loyalty offers to preserve recurring revenue.

---

## 🚀 Key Features

- **End-to-End ML Pipeline**: Runs data cleaning, column transformation, class imbalance resolution, model comparison, and hyperparameter optimization.
- **Advanced Imbalance Handling**: Utilizes **SMOTEENN** (SMOTE + Edited Nearest Neighbors) to balance minority churn classes for higher sensitivity.
- **Multi-Model Evaluator**: Trains and registers baseline metrics for six classifiers:
  1. *Logistic Regression*
  2. *Decision Tree*
  3. *Random Forest* (Optuna optimized)
  4. *XGBoost*
  5. *LightGBM*
  6. *CatBoost*
- **Hyperparameter Optimization**: Uses **Optuna** to perform Bayesian optimization on the top classifier.
- **Interactive Streamlit Web Dashboard**:
  - **Single Customer Predictor**: Input profile elements in real-time, compute a churn probability gauge, and get customized retention recommendations.
  - **Batch Predictor (CSV Upload)**: Upload a CSV containing multiple records (generates identifiers automatically if missing), get bulk churn rates, and download predicted results.
  - **Model Insights Dashboard**: View confusion matrix, ROC curves, baseline comparisons, and feature importances.
  - **EDA Tab**: Dynamic statistical data exploration, tenure distributions, and correlation maps.
- **Dockerized AWS EC2 Deployment**: Fully containerized and optimized to run on the **AWS EC2 Free Tier** (`t2.micro`).

---

## 📁 Project Structure

```text
telecom/
├── app/
│   └── app.py                  # Streamlit Web Application (Frontend Dashboard)
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Raw Dataset (Git-ignored)
├── models/                     # Saved Models & Validation Charts
│   ├── best_churn_model.joblib # Serialized fitted preprocessor & optimized model
│   ├── metrics.json            # Model performance logs & metrics
│   ├── confusion_matrix.png    # Validation Confusion Matrix
│   ├── roc_curve.png           # Validation ROC Curve
│   └── feature_importance.png  # Top drivers of churn chart
├── src/                        # Machine Learning Pipeline Source Code
│   ├── download_data.py        # Dataset Downloader
│   ├── data_preprocessing.py   # Data Cleaning & Column Transformer Pipeline
│   └── train.py                # Pipeline Comparison, Optuna Tuning, & Evaluation
├── Dockerfile                  # Containerization Blueprint
├── .dockerignore               # Files to ignore during Docker builds
├── .gitignore                  # Files to ignore in Git version control
├── requirements.txt            # Project Dependencies
└── deploy_ec2.sh               # AWS EC2 Automated Deployment Script
```

---

## 🛠️ Local Installation & Setup

### 1. Clone & Navigate to Workspace
```powershell
git clone https://github.com/prakashkmb12-afk/Telecom-churn-prediction.git
cd Telecom-churn-prediction
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

## 🏃 Running the Pipeline Locally

### Step 1: Download the Dataset
The dataset will be downloaded programmatically from the IBM database:
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
This opens the application in your default browser at `http://localhost:8501`.

---

## ☁️ Deploying to AWS EC2 (Free Tier)

This project contains Docker setup configurations designed to deploy the app on AWS EC2 Free Tier.

### Step 1: Launch an EC2 Instance on AWS
1. Open the **AWS EC2 Console** in region `ap-south-1` (Mumbai).
2. Launch a new instance with the **Ubuntu 22.04 LTS** OS (Free tier eligible).
3. Select **`t2.micro`** as the instance type (Free tier eligible).
4. Create and download your SSH key pair (`telecom-key.pem`).
5. Set storage to **20 GB gp3** (under the 30 GB Free Tier limit).
6. Edit the **Security Group** and add a **Custom TCP Inbound Rule** allowing port **`8501`** from **Anywhere-IPv4 (`0.0.0.0/0`)**.

### Step 2: Connect and Run the Deploy Script
Log into your EC2 instance via SSH or EC2 Instance Connect and execute:
```bash
# Clone the repository
git clone https://github.com/prakashkmb12-afk/Telecom-churn-prediction.git
cd Telecom-churn-prediction

# Make the script executable and run it
chmod +x deploy_ec2.sh
./deploy_ec2.sh
```
Once completed, you can access your live application at `http://<your-ec2-public-ip>:8501`.
