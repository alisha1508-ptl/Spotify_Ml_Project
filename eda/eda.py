# eda.py
import pandas as pd

# -----------------------------
# Absolute path for dataset
# -----------------------------
data_path = r"D:\spotify_ml_project\spotify_ml_project\data\clean_dataset.csv"

# Load dataset
df = pd.read_csv(data_path)

# -----------------------------
# 1️⃣ Data Preview
# -----------------------------
print("=== First 5 Rows ===")
print(df.head())

print("\n=== Columns & Types ===")
print(df.info())

# -----------------------------
# 2️⃣ Descriptive Statistics
# -----------------------------
print("\n=== Descriptive Statistics ===")
print(df.describe())

# -----------------------------
# 3️⃣ Missing Values
# -----------------------------
print("\n=== Missing Values ===")
print(df.isnull().sum())