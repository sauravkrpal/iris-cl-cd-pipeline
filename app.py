import pickle
from flask import Flask, request, jsonify

# Load model
with open("iris_model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route("/")
def home():
    return "ML Model Deployment with Flask + Render!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = data.get("features")  # Expect list of features
    prediction = model.predict([features])
    return jsonify({"prediction": prediction.tolist()})

if __name__ == "__main__":
    app.run()
