"""Flask web service that serves the churn model (ML Zoomcamp 5.3 / 5.4)."""
import pickle

from flask import Flask, jsonify, request

MODEL_FILE = "model_C=1.0.bin"

with open(MODEL_FILE, "rb") as f_in:
    dv, model = pickle.load(f_in)

# Elastic Beanstalk looks for a variable called "application" by default
# when using the Python platform, so we expose both names.
app = Flask("churn")
application = app


@app.route("/", methods=["GET"])
def index():
    return jsonify({"service": "churn-prediction", "status": "ok"})


@app.route("/ping", methods=["GET"])
def ping():
    return "PONG"


@app.route("/predict", methods=["POST"])
def predict():
    customer = request.get_json()
    X = dv.transform([customer])
    y_pred = float(model.predict_proba(X)[0, 1])
    churn = y_pred >= 0.5

    return jsonify({"churn_probability": y_pred, "churn": bool(churn)})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
