import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
DATA_PATH = r"C:\Users\disha\Desktop\spotify_ml_project\data\clean_dataset.csv"

df = pd.read_csv(DATA_PATH)

# Create target column
df['popular'] = (df['popularity'] > 60).astype(int)

# Features
features = [
'danceability','energy','key','loudness','mode',
'speechiness','acousticness','instrumentalness',
'liveness','valence','tempo','duration_ms'
]

X = df[features]
y = df['popular']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# Save model
MODEL_PATH = r"C:\Users\disha\Desktop\spotify_ml_project\model.pkl"

pickle.dump(model, open(MODEL_PATH, "wb"))

print("Model saved successfully!")