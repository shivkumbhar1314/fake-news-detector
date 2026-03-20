import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)

# Load trained ML model
model = pickle.load(open("model/fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return "Fake News Detection API Running"

@app.route("/predict", methods=["POST"])
def predict_news():

    data = request.json
    news = data["news"]

    vect = vectorizer.transform([news])

    prediction = model.predict(vect)[0]

    if prediction == 1:
        result = "Fake News"
    else:
        result = "Real News"

    return jsonify({
        "prediction": result
    })

if __name__ == "__main__":
    print("Server running on http://127.0.0.1:5000")
    app.run(debug=True)