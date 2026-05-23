import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Set page config
st.set_page_config(
    page_title="Telecom Churn Prediction System",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    /* Gradient Background & Sleek Fonts */
    .reportview-container {
        background: #0f172a;
    }
    
    /* Premium Header style */
    .main-header {
        font-family: 'Outfit', sans-serif;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Card Glassmorphism effect */
    .premium-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }
    
    /* Metrics box */
    .churn-box {
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        font-weight: bold;
        font-size: 1.5rem;
        margin-top: 15px;
    }
    
    .churn-yes {
        background-color: rgba(239, 68, 68, 0.15);
        border: 2px solid #ef4444;
        color: #f87171;
    }
    
    .churn-no {
        background-color: rgba(34, 197, 94, 0.15);
        border: 2px solid #22c55e;
        color: #4ade80;
    }
    
    /* Recommendation Card styling */
    .rec-card {
        background: rgba(59, 130, 246, 0.1);
        border-left: 5px solid #3b82f6;
        border-radius: 4px;
        padding: 15px;
        margin-top: 15px;
        color: #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# Helper to load model
@st.cache_resource
def load_churn_model():
    # Check possible paths
    paths = [
        "models/best_churn_model.joblib",
        "../models/best_churn_model.joblib",
        "d:/Projects/telecom/models/best_churn_model.joblib"
    ]
    for path in paths:
        if os.path.exists(path):
            try:
                model_data = joblib.load(path)
                return model_data
            except Exception as e:
                st.error(f"Error loading model from {path}: {e}")
    return None

# Load dataset for explorer and basic statistics
@st.cache_data
def load_raw_data():
    paths = [
        "data/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        "../data/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        "d:/Projects/telecom/data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    ]
    for path in paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                # Quick clean
                df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0.0)
                df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
                return df
            except Exception as e:
                st.error(f"Error loading data from {path}: {e}")
    return None

model_data = load_churn_model()
df_raw = load_raw_data()

# App Sidebar Header
st.sidebar.markdown("<div style='text-align: center; margin-bottom: 20px;'><h2 style='color:#3b82f6;'>📞 Telco Churn AI</h2></div>", unsafe_allow_html=True)
app_mode = st.sidebar.selectbox("Navigate System", ["Single Customer Predictor", "Batch Predictor (CSV Upload)", "Model Insights & Performance", "Exploratory Data Analysis (EDA)"])

if model_data is None:
    st.error("⚠️ Prediction model not found! Please run the training script first to train the machine learning pipeline.")
    st.info("💡 To train the model, run `python src/train.py` from the project directory.")
else:
    preprocessor = model_data["preprocessor"]
    model = model_data["model"]
    model_name = model_data["model_name"]

# ==========================================
# MODE 1: SINGLE CUSTOMER PREDICTOR
# ==========================================
if app_mode == "Single Customer Predictor":
    st.markdown("<h1 class='main-header'>Telecom Customer Churn Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Enter customer details below to predict their likelihood of leaving the service in real-time.</p>", unsafe_allow_html=True)
    
    if model_data is not None:
        st.write(f"ℹ️ Currently running optimized **{model_name}** pipeline.")
        
        # Create columns for input categorization
        tab_demo, tab_service, tab_billing = st.tabs(["👤 Demographic Profile", "🔌 Subscribed Services", "💳 Contract & Billing"])
        
        with tab_demo:
            col1, col2 = st.columns(2)
            with col1:
                gender = st.selectbox("Gender", ["Female", "Male"])
                senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            with col2:
                partner = st.selectbox("Has Partner?", ["Yes", "No"])
                dependents = st.selectbox("Has Dependents?", ["No", "Yes"])
                
        with tab_service:
            col1, col2, col3 = st.columns(3)
            with col1:
                phone_service = st.selectbox("Phone Service", ["Yes", "No"])
                multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"] if phone_service == "Yes" else ["No phone service"])
                internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
            with col2:
                online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"] if internet_service != "No" else ["No internet service"])
                online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"] if internet_service != "No" else ["No internet service"])
                device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"] if internet_service != "No" else ["No internet service"])
            with col3:
                tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"] if internet_service != "No" else ["No internet service"])
                streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"] if internet_service != "No" else ["No internet service"])
                streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"] if internet_service != "No" else ["No internet service"])
                
        with tab_billing:
            col1, col2 = st.columns(2)
            with col1:
                contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
                paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
                payment_method = st.selectbox("Payment Method", [
                    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
                ])
            with col2:
                tenure = st.slider("Tenure (Months active)", min_value=0, max_value=72, value=12)
                monthly_charges = st.slider("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=70.0)
                
                # Dynamic/Estimated Total Charges based on tenure & monthly charges
                estimated_total = tenure * monthly_charges
                total_charges = st.number_input(
                    "Total Charges ($)", 
                    min_value=0.0, 
                    max_value=9000.0, 
                    value=float(estimated_total)
                )

        # Single prediction button
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔮 Predict Churn Probability", use_container_width=True):
            # Create a dictionary matches the exact feature name expected
            input_dict = {
                "gender": gender,
                "SeniorCitizen": senior_citizen,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless_billing,
                "PaymentMethod": payment_method,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges
            }
            
            # Convert to DataFrame
            input_df = pd.DataFrame([input_dict])
            
            # Predict
            try:
                # Preprocess and predict
                processed_input = preprocessor.transform(input_df)
                probability = model.predict_proba(processed_input)[0][1]
                prediction = int(probability >= 0.5)
                
                st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
                st.subheader("📊 Prediction Results")
                
                col_left, col_right = st.columns(2)
                
                with col_left:
                    # Churn probability meter
                    st.metric(label="Churn Probability", value=f"{probability*100:.1f}%")
                    st.progress(float(probability))
                    
                    if prediction == 1:
                        st.markdown(
                            "<div class='churn-box churn-yes'>⚠️ HIGH RISK OF CHURN</div>", 
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            "<div class='churn-box churn-no'>✅ LOW RISK / LOYAL CUSTOMER</div>", 
                            unsafe_allow_html=True
                        )
                        
                with col_right:
                    st.subheader("💡 Actionable Retention Strategy")
                    recs = []
                    
                    if prediction == 1:
                        recs.append("🔴 **High Churn Risk Detected.** Action is recommended immediately.")
                        if contract == "Month-to-month":
                            recs.append("👉 **Contract Upgrade Plan:** Offer a 1-year contract with a discounted rate (e.g. 15% discount) to lock in loyalty.")
                        if internet_service == "Fiber optic" and tech_support == "No":
                            recs.append("👉 **Technical Onboarding:** High charges for Fiber optic without tech support usually lead to frustration. Offer a free month of premium Tech Support.")
                        if monthly_charges > 80.0:
                            recs.append("👉 **Plan Optimization:** Recommend downgrading non-essential add-ons or matching competitor pricing to lower monthly charges.")
                        if payment_method == "Electronic check":
                            recs.append("👉 **Billing Incentive:** Offer a one-time $10 credit to set up automatic payment (credit card or bank transfer) for smoother billing.")
                    else:
                        recs.append("🟢 **Customer is likely to stay.** Keep doing what you're doing!")
                        if contract != "Two year":
                            recs.append("👉 **Upsell Loyalty:** Customer is stable. Target them with multi-year contract benefits or bundled family plans for long-term customer value.")
                            
                    st.markdown("\n".join([f"<p style='color:#e2e8f0;'>{r}</p>" for r in recs]), unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error running model prediction: {e}")

# ==========================================
# MODE 2: BATCH PREDICTOR
# ==========================================
elif app_mode == "Batch Predictor (CSV Upload)":
    st.markdown("<h1 class='main-header'>Batch Customer Churn Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Upload a CSV file containing multiple customer records to predict churn probabilities in bulk.</p>", unsafe_allow_html=True)
    
    st.markdown("""
    ### 📂 Instructions:
    Upload a CSV file containing customer details. The file **must** include the following features:
    `gender`, `SeniorCitizen` (Yes/No or 0/1), `Partner`, `Dependents`, `tenure`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`.
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
            st.dataframe(input_df.head())
            
            # Clean data (duplicate handling and data types)
            df_cleaned = input_df.copy()
            if 'customerID' in df_cleaned.columns:
                customer_ids = df_cleaned['customerID']
                df_cleaned = df_cleaned.drop(columns=['customerID'])
            else:
                customer_ids = pd.Series([f"CUST-{i}" for i in range(len(df_cleaned))])
                
            df_cleaned['TotalCharges'] = pd.to_numeric(df_cleaned['TotalCharges'], errors='coerce').fillna(0.0)
            df_cleaned['SeniorCitizen'] = df_cleaned['SeniorCitizen'].map({0: 'No', 1: 'Yes', '0': 'No', '1': 'Yes'}).fillna(df_cleaned['SeniorCitizen'])
            
            # Run prediction
            processed_data = preprocessor.transform(df_cleaned)
            probabilities = model.predict_proba(processed_data)[:, 1]
            predictions = (probabilities >= 0.5).astype(int)
            
            # Create results dataframe
            results_df = input_df.copy()
            if "customerID" not in results_df.columns:
                results_df["customerID"] = customer_ids
            results_df["Churn_Probability"] = probabilities
            results_df["Churn_Prediction"] = ["Yes" if p == 1 else "No" for p in predictions]
            
            st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
            st.subheader("📊 Batch Prediction Results")
            
            # Summarize results
            total_customers = len(results_df)
            churn_count = sum(predictions)
            non_churn_count = total_customers - churn_count
            churn_rate = churn_count / total_customers
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Customers", total_customers)
            col2.metric("Predicted Churn", f"{churn_count} ({churn_rate*100:.1f}%)")
            col3.metric("Predicted Stay", non_churn_count)
            
            st.dataframe(results_df[["customerID", "Churn_Probability", "Churn_Prediction"] + list(df_cleaned.columns[:4])].head(20))
            
            # Allow download
            csv_data = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Complete Prediction CSV",
                csv_data,
                "telco_churn_predictions.csv",
                "text/csv",
                key='download-csv'
            )
            st.markdown("</div>", unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error processing CSV or predicting: {e}")

# ==========================================
# MODE 3: MODEL INSIGHTS
# ==========================================
elif app_mode == "Model Insights & Performance":
    st.markdown("<h1 class='main-header'>Model Training Insights & Metrics</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Review performance metrics, confusion matrix, ROC curves, and top predictors of churn.</p>", unsafe_allow_html=True)
    
    # Load training metrics
    metrics_path = "models/metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics_data = json.load(f)
            
        final_metrics = metrics_data["final_metrics"]
        st.subheader("📈 Overall Model Performance")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Accuracy", f"{final_metrics['accuracy']*100:.2f}%")
        col2.metric("Precision", f"{final_metrics['precision']*100:.2f}%")
        col3.metric("Recall (Sensitivity)", f"{final_metrics['recall']*100:.2f}%")
        col4.metric("F1-Score", f"{final_metrics['f1_score']*100:.2f}%")
        col5.metric("ROC-AUC", f"{final_metrics['roc_auc']:.4f}")
        
        # Display baseline comparison table
        st.write("---")
        st.subheader("⚖️ Baseline Models Comparison")
        comp_df = pd.DataFrame(metrics_data["baseline_comparison"]).T
        st.dataframe(comp_df.style.highlight_max(axis=0, color='rgba(59, 130, 246, 0.2)'))
        
        # Display images of evaluations
        st.write("---")
        st.subheader("🖼️ Model Evaluation Visualizations")
        col_img1, col_img2 = st.columns(2)
        
        cm_path = "models/confusion_matrix.png"
        roc_path = "models/roc_curve.png"
        feat_path = "models/feature_importance.png"
        
        if os.path.exists(cm_path):
            col_img1.image(cm_path, caption="Confusion Matrix (Evaluating True vs. False Predictions)")
        if os.path.exists(roc_path):
            col_img2.image(roc_path, caption="ROC Curve (True Positive vs. False Positive Rates)")
            
        st.write("---")
        if os.path.exists(feat_path):
            st.image(feat_path, caption="Top Feature Importances (Key Drivers of Customer Churn)")
    else:
        st.warning("Training metrics and visualization charts not found under `models/` directory. Run `python src/train.py` first to generate model performance charts.")

# ==========================================
# MODE 4: EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
elif app_mode == "Exploratory Data Analysis (EDA)":
    st.markdown("<h1 class='main-header'>Exploratory Data Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Inspect statistical profiles, distributions, and demographic correlations with churn rates.</p>", unsafe_allow_html=True)
    
    if df_raw is not None:
        st.subheader("📋 Dataset Quick Overview")
        st.dataframe(df_raw.head())
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Customers in Dataset", len(df_raw))
        
        raw_churn_rate = df_raw['Churn'].value_counts(normalize=True).get('Yes', 0)
        col2.metric("Overall Churn Rate", f"{raw_churn_rate*100:.2f}%")
        
        avg_monthly = df_raw['MonthlyCharges'].mean()
        col3.metric("Average Monthly Charge", f"${avg_monthly:.2f}")
        
        st.write("---")
        st.subheader("📊 Visualizing Key Churn Drivers")
        
        # Custom EDA Plots
        eda_metric = st.selectbox("Select Feature to Analyze Churn Correlation", ["Contract", "InternetService", "PaymentMethod", "SeniorCitizen", "Partner"])
        
        fig, ax = plt.subplots(figsize=(10, 5))
        # Compute churn percentage per group
        churn_by_group = df_raw.groupby(eda_metric)['Churn'].value_counts(normalize=True).unstack() * 100
        churn_by_group.plot(kind='bar', stacked=True, color=['#4ade80', '#f87171'], ax=ax)
        
        ax.set_ylabel("Percentage (%)")
        ax.set_title(f"Churn Distribution by {eda_metric}")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Charges vs. Tenure plot
        st.write("---")
        st.subheader("📈 Monthly Charges vs. Customer Tenure (colored by Churn)")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            data=df_raw.sample(min(1500, len(df_raw))), 
            x='tenure', 
            y='MonthlyCharges', 
            hue='Churn', 
            palette={'Yes': '#ef4444', 'No': '#22c55e'},
            alpha=0.6,
            ax=ax2
        )
        ax2.set_xlabel("Tenure (Months)")
        ax2.set_ylabel("Monthly Charges ($)")
        ax2.set_title("Customer Segmentation: Tenure vs Monthly Charges")
        plt.tight_layout()
        st.pyplot(fig2)
        
    else:
        st.warning("Telecom customer dataset not found. Download the dataset first.")
