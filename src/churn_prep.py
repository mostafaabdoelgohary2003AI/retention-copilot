"""Data loading, cleaning and feature engineering for the Iranian churn dataset (UCI #563)."""
import pandas as pd
import numpy as np

RAW = "data/Customer_Churn.csv"
TARGET = "Churn"
RENAME = {
    "Call  Failure": "call_failures",
    "Complains": "complaints",
    "Subscription  Length": "subscription_length",
    "Charge  Amount": "charge_amount",
    "Seconds of Use": "seconds_of_use",
    "Frequency of use": "frequency_of_use",
    "Frequency of SMS": "frequency_of_sms",
    "Distinct Called Numbers": "distinct_called_numbers",
    "Age Group": "age_group",
    "Tariff Plan": "tariff_plan",
    "Status": "status",
    "Age": "age",
    "Customer Value": "customer_value",
}
FEATURES = list(RENAME.values()) + ["calls_per_month", "sms_ratio"]

def load_raw(path=RAW):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns=RENAME)
    df = df.dropna().reset_index(drop=True)
    return df

def engineer(df):
    df = df.copy()
    df["calls_per_month"] = df["frequency_of_use"] / df["subscription_length"].clip(lower=1)
    df["sms_ratio"] = df["frequency_of_sms"] / (df["frequency_of_use"] + 1.0)
    return df

def load_xy(path=RAW):
    df = engineer(load_raw(path))
    y = df[TARGET].astype(int)
    X = df[FEATURES]
    return X, y
