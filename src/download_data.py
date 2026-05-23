import os
import urllib.request

def download_dataset():
    url = "https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    
    print(f"Downloading dataset from {url}...")
    try:
        urllib.request.urlretrieve(url, filepath)
        print(f"Dataset successfully downloaded and saved to {filepath}")
    except Exception as e:
        print(f"Error downloading dataset: {e}")

if __name__ == "__main__":
    download_dataset()
