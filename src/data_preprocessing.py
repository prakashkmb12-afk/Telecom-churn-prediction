import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import os

def load_and_clean_data(filepath="data/WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    """
    Load the telecom customer churn dataset and perform initial data cleaning.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    
    # Load dataset
    df = pd.read_csv(filepath)
    
    # Drop customerID as it is unique and does not contribute to modeling
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
        
    # TotalCharges contains empty spaces for customers with 0 tenure.
    # Convert TotalCharges to numeric, coerce errors to NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # For tenure = 0, TotalCharges is NaN. Fill with 0.0 since they haven't been billed yet
    df['TotalCharges'] = df['TotalCharges'].fillna(0.0)
    
    # Convert SeniorCitizen to string category so it can be handled by OneHotEncoder
    df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
    
    # Convert target Churn to binary numeric (1 for Yes, 0 for No)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    return df

def get_preprocessor():
    """
    Create a scikit-learn ColumnTransformer for preprocessing numerical and categorical features.
    """
    # Define features based on their type
    categorical_cols = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]
    
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    # Preprocessing pipelines for both numerical and categorical data
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    return preprocessor, numerical_cols, categorical_cols

def prepare_data(filepath="data/WA_Fn-UseC_-Telco-Customer-Churn.csv", test_size=0.2, random_state=42):
    """
    Load data, clean it, split into train and test sets, and return splits.
    """
    df = load_and_clean_data(filepath)
    
    # Separate features and target
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    print("Testing data cleaning and preparation...")
    X_train, X_test, y_train, y_test = prepare_data()
    preprocessor, num_cols, cat_cols = get_preprocessor()
    
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train class distribution: \n{y_train.value_counts(normalize=True)}")
    print("Preproccessing setup complete!")
