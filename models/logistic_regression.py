import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -------------------------------
# Load dataset
df = pd.read_csv("data/clean_dataset.csv")

# -------------------------------
# Convert popularity into binary
df['popularity'] = df['popularity'].apply(lambda x: 1 if x > 30 else 0)

# -------------------------------
# Features
features = [
'danceability','energy','key','loudness','mode',
'speechiness','acousticness','instrumentalness',
'liveness','valence','tempo','duration_ms'
]

X = df[features]

# Target
y = df['popularity']

# -------------------------------
# Split data
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Train model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# -------------------------------
# Prediction
y_pred = model.predict(X_test)

# -------------------------------
# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -------------------------------
# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("\n✅ Model saved as model.pkl")