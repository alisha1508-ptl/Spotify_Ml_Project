import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/clean_dataset.csv")

# -------------------------------
# 1. Popularity Distribution
# -------------------------------
plt.figure(figsize=(6,4))
sns.histplot(df['popularity'], bins=30, kde=True)
plt.title("Popularity Distribution")
plt.xlabel("Popularity")
plt.ylabel("Count")

plt.savefig("visualization/popularity_distribution.png")
plt.show()


# -------------------------------
# 2. Danceability vs Popularity
# -------------------------------
plt.figure(figsize=(6,4))
sns.scatterplot(x=df['danceability'], y=df['popularity'])
plt.title("Danceability vs Popularity")

plt.savefig("visualization/danceability_vs_popularity.png")
plt.show()


# -------------------------------
# 3. Energy vs Popularity
# -------------------------------
plt.figure(figsize=(6,4))
sns.scatterplot(x=df['energy'], y=df['popularity'])
plt.title("Energy vs Popularity")

plt.savefig("visualization/energy_vs_popularity.png")
plt.show()


# -------------------------------
# 4. Tempo Distribution
# -------------------------------
plt.figure(figsize=(6,4))
sns.histplot(df['tempo'], bins=30, kde=True)
plt.title("Tempo Distribution")

plt.savefig("visualization/tempo_distribution.png")
plt.show()


# -------------------------------
# 5. Correlation Heatmap
# -------------------------------
numeric_df = df.select_dtypes(include='number')

plt.figure(figsize=(10,8))
sns.heatmap(numeric_df.corr(), cmap="coolwarm", annot=True)

plt.title("Correlation Heatmap")

plt.savefig("visualization/correlation_heatmap.png")
plt.show()