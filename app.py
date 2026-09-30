import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

bundle = joblib.load("diabetes_model.pkl")
model = bundle["model"]
FEATURES = bundle["features"]
DEFAULTS = bundle["defaults"]
MODEL_NAME = bundle["name"]


@app.route("/")
def index():
    return render_template("index.html", features=FEATURES,
                           defaults=DEFAULTS, model_name=MODEL_NAME)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        values = {f: float(request.form[f]) for f in FEATURES}
    except (KeyError, ValueError):
        return render_template("index.html", features=FEATURES, defaults=DEFAULTS,
                               model_name=MODEL_NAME,
                               error="Veuillez saisir des valeurs numeriques valides.")

    X = pd.DataFrame([values], columns=FEATURES)
    pred = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0][1]) * 100

    return render_template("result.html", values=values, prediction=pred,
                           proba=round(proba, 1), model_name=MODEL_NAME)


if __name__ == "__main__":
    app.run(debug=True)
