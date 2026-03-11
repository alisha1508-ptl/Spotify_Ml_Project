# -------------------------------
# eda_dashboard.py

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -------------------------------
# Page config
st.set_page_config(page_title="Spotify ML Dashboard", layout="wide")

# -------------------------------
# Dark Mode Toggle
dark_mode = st.sidebar.checkbox("🌙 Dark Mode")

if dark_mode:
    bg_color = "#121212"
    text_color = "white"
    plt_style = "dark_background"
else:
    bg_color = "#f0f2f6"
    text_color = "#000000"
    plt_style = "default"

plt.style.use(plt_style)

# -------------------------------
# Custom CSS
st.markdown(f"""
<style>

.stApp {{
background-color:{bg_color};
color:{text_color};
}}

.main-header {{
color:#1db954;
font-size:38px;
font-weight:bold;
text-align:center;
}}

footer {{
visibility:hidden;
}}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# Title
st.markdown('<h1 class="main-header">🎵 Spotify Machine Learning Dashboard</h1>', unsafe_allow_html=True)

# -------------------------------
# Load Dataset
DATA_PATH = r"D:\spotify_ml_project\spotify_ml_project\data\clean_dataset.csv"

df = pd.read_csv(DATA_PATH)

# ==========================================================
# 1️⃣ EDA
# ==========================================================

st.header("📊 Exploratory Data Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

with col2:
    st.subheader("Statistics")
    st.dataframe(df.describe())

st.subheader("Missing Values")
st.dataframe(df.isnull().sum())

# ==========================================================
# 2️⃣ EDA VISUALIZATION
# ==========================================================

st.header("📈 EDA Visualizations")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(5,4))
    df['track_genre'].value_counts().head(10).plot.bar(ax=ax)
    ax.set_title("Top Genres")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(5,4))
    sns.histplot(df['danceability'], bins=20, ax=ax)
    ax.set_title("Danceability Distribution")
    st.pyplot(fig)

st.subheader("Correlation Heatmap")

fig, ax = plt.subplots(figsize=(7,5))
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", ax=ax)
st.pyplot(fig)

# ==========================================================
# 3️⃣ SIMPLE LINEAR REGRESSION
# ==========================================================

st.header("📉 Simple Linear Regression")

X_simple = df[['danceability']]
y = df['popularity']

model_simple = LinearRegression()
model_simple.fit(X_simple, y)

fig, ax = plt.subplots(figsize=(5,4))
sns.regplot(x="danceability", y="popularity", data=df, ax=ax)

st.pyplot(fig)

st.subheader("🎯 Live Prediction (Simple Linear Regression)")

dance_input = st.slider("Select Danceability",0.0,1.0,0.5,key="s1")

simple_input = pd.DataFrame([[dance_input]],columns=['danceability'])

simple_prediction = model_simple.predict(simple_input)[0]

st.success(f"Predicted Popularity: {round(simple_prediction,2)}")

# ==========================================================
# 4️⃣ MULTIPLE LINEAR REGRESSION
# ==========================================================

st.header("📊 Multiple Linear Regression")

features_multi = ['danceability','energy','loudness','tempo','valence']

X_multi = df[features_multi]
y_multi = df['popularity']

model_multi = LinearRegression()
model_multi.fit(X_multi, y_multi)

st.subheader("🎯 Live Prediction (Multiple Linear Regression)")

dance_m = st.slider("Danceability",0.0,1.0,0.5,key="m1")
energy_m = st.slider("Energy",0.0,1.0,0.5,key="m2")
loudness_m = st.slider("Loudness",-60.0,0.0,-10.0,key="m3")
tempo_m = st.slider("Tempo",0.0,250.0,120.0,key="m4")
valence_m = st.slider("Valence",0.0,1.0,0.5,key="m5")

multi_input = pd.DataFrame([[dance_m,energy_m,loudness_m,tempo_m,valence_m]],columns=features_multi)

multi_prediction = model_multi.predict(multi_input)[0]

st.success(f"Predicted Popularity (Multiple Regression): {round(multi_prediction,2)}")

# ==========================================================
# 5️⃣ LOGISTIC REGRESSION EXPLANATION
# ==========================================================

st.header("📘 Logistic Regression Explanation")

st.markdown("""
Logistic Regression is a **classification algorithm** used to predict categories.

In this project we classify songs into:

- **Popular (1)**
- **Not Popular (0)**

The model uses audio features such as:

- Danceability
- Energy
- Loudness
- Tempo
- Valence
- Acousticness
- Instrumentalness

The output is calculated using the **Sigmoid Function**:

P = 1 / (1 + e^-z)

If probability > 0.5 → Song is **Popular**

If probability < 0.5 → Song is **Not Popular**
""")

# ==========================================================
# 6️⃣ SIGMOID CURVE VISUALIZATION
# ==========================================================

st.header("📈 Sigmoid Curve Visualization")

z = np.linspace(-10,10,100)
sigmoid = 1/(1+np.exp(-z))

fig, ax = plt.subplots()

ax.plot(z, sigmoid)

ax.set_title("Sigmoid Function")
ax.set_xlabel("Z value")
ax.set_ylabel("Probability")

ax.axhline(0.5, linestyle="--")
ax.axvline(0, linestyle="--")

st.pyplot(fig)

# ==========================================================
# 7️⃣ LOGISTIC REGRESSION MODEL
# ==========================================================

st.header("🤖 Logistic Regression Model")

df['popular'] = (df['popularity'] > 60).astype(int)

features = [
'danceability','energy','key','loudness','mode',
'speechiness','acousticness','instrumentalness',
'liveness','valence','tempo','duration_ms'
]

X = df[features]
y_class = df['popular']

X_train, X_test, y_train, y_test = train_test_split(
X, y_class, test_size=0.2, random_state=42
)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

pred = log_model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

cm = confusion_matrix(y_test, pred)

report = classification_report(y_test, pred, zero_division=0)

# ==========================================================
# 8️⃣ MODEL EVALUATION
# ==========================================================

st.header("📊 Model Evaluation")

st.success(f"Model Accuracy: {round(accuracy*100,2)} %")

st.subheader("Confusion Matrix")

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)

st.subheader("Classification Report")
st.text(report)

# ==========================================================
# 9️⃣ LOGISTIC PREDICTION
# ==========================================================

st.header("🎯 Song Popularity Prediction")

danceability = st.slider("Danceability",0.0,1.0,0.5,key="l1")
energy = st.slider("Energy",0.0,1.0,0.5,key="l2")
key = st.slider("Key",0,11,5,key="l3")
loudness = st.slider("Loudness",-60.0,0.0,-10.0,key="l4")
mode = st.selectbox("Mode",[0,1],key="l5")
speechiness = st.slider("Speechiness",0.0,1.0,0.05,key="l6")
acousticness = st.slider("Acousticness",0.0,1.0,0.1,key="l7")
instrumentalness = st.slider("Instrumentalness",0.0,1.0,0.0,key="l8")
liveness = st.slider("Liveness",0.0,1.0,0.1,key="l9")
valence = st.slider("Valence",0.0,1.0,0.5,key="l10")
tempo = st.slider("Tempo",0.0,250.0,120.0,key="l11")
duration_ms = st.number_input("Duration (ms)",0,600000,200000,key="l12")

if st.button("Predict Popularity"):

    input_data = [[
        danceability,energy,key,loudness,mode,
        speechiness,acousticness,instrumentalness,
        liveness,valence,tempo,duration_ms
    ]]

    input_df = pd.DataFrame(input_data, columns=features)

    prediction = log_model.predict(input_df)[0]
    probability = log_model.predict_proba(input_df)[0][1]

    st.progress(int(probability*100))
    st.write("Chance of being popular:", round(probability*100,2), "%")

    if prediction == 1:
        st.success("🎵 This Song Will Be Popular")
    else:
        st.error("🎵 This Song Will Not Be Popular")