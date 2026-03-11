# eda_visualization.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Absolute path for dataset
# -----------------------------
data_path = r"D:\spotify_ml_project\spotify_ml_project\data\clean_dataset.csv"


# Load dataset
df = pd.read_csv(data_path)

# -----------------------------
# 1️⃣ Top 10 Numeric Columns Histograms
# -----------------------------
num_cols = df.select_dtypes(include='number').columns[:10]
df[num_cols].hist(figsize=(15, 10), bins=20, color='skyblue')
plt.suptitle("Histograms of Top 10 Numeric Columns")
plt.show()

# -----------------------------
# 2️⃣ Pie Chart Example
# -----------------------------
if 'genre' in df.columns:
    plt.figure(figsize=(6,6))
    df['genre'].value_counts().plot.pie(
        autopct='%1.1f%%', colors=sns.color_palette('pastel')
    )
    plt.title('Genre Distribution')
    plt.ylabel('')
    plt.show()

# -----------------------------
# 3️⃣ Bar Chart Example
# -----------------------------
if 'artist_name' in df.columns:
    plt.figure(figsize=(8,5))
    df['artist_name'].value_counts().head(5).plot.bar(color='orange')
    plt.title('Top 5 Artists')
    plt.xlabel('Artist')
    plt.ylabel('Count')
    plt.show()

# -----------------------------
# 4️⃣ Correlation Heatmap
# -----------------------------
plt.figure(figsize=(10,8))
sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()