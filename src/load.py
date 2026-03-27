import pandas as pd

def load_data(path="data/telco_churn.csv"):
    df = pd.read_csv(path)
    return df