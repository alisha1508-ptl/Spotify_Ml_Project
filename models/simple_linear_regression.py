import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/clean_dataset.csv")

# Feature (X) and Target (y)
X = df[['danceability']]   # independent variable
y = df['energy']           # dependent variable

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model create
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Print model details
print("Coefficient:", model.coef_)
print("Intercept:", model.intercept_)

# Visualization
plt.scatter(X_test, y_test, label="Actual Data")
plt.plot(X_test, y_pred, label="Regression Line")

plt.xlabel("Danceability")
plt.ylabel("Energy")
plt.title("Simple Linear Regression")

plt.legend()
plt.show()