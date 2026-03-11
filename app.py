from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    # Collect form values
    form_values = list(request.form.values())

    # Replace empty strings with 0 and convert to float
    features = [float(x) if x != "" else 0 for x in form_values]

    # Make sure the number of features matches the model
    expected_features = model.coef_.shape[1]  # number of features model expects

    if len(features) != expected_features:
        return f"Error: Model expects {expected_features} features, but got {len(features)}."

    prediction = model.predict([features])

    if prediction[0] == 1:
        result = "Song will be Popular"
    else:
        result = "Song may not be Popular"

    return render_template("index.html", prediction_text=result)