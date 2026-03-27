import pandas as pd

def preprocess(df):
    df = df.copy()

    # target
    df["Churn"] = df["Churn"].map({"Churned": 1, "Stayed": 0})

    # remove ID
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # fix TotalCharges
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # encoding
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    print("Rows after preprocessing:", len(df))

    return X, y