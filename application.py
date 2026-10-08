import os
import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application
app.url_map.strict_slashes = False

## Import ridge regressor and standard scaler pickle
model_path = 'models/Ridge.pk1' if os.path.exists('models/Ridge.pk1') else 'models/ridge.pk1'
scaler_path = 'models/scaler.pk1' if os.path.exists('models/scaler.pk1') else 'models/Scaler.pk1'

ridge_model = pickle.load(open(model_path, 'rb'))
standard_scaler = pickle.load(open(scaler_path, 'rb'))


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/predictdata", methods=["GET", "POST"])
@app.route("/predict", methods=["GET", "POST"])
@app.route("/prediction", methods=["GET", "POST"])
@app.route("/home", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "POST":
        Temperature = float(request.form.get(" Temperature"))
        RH = float(request.form.get("RH"))
        Ws = float(request.form.get("Ws"))
        Rain = float(request.form.get("Rain"))
        FFMC = float(request.form.get("FFMC"))
        DMC = float(request.form.get("DMC"))
        ISI = float(request.form.get("ISI"))
        Classes = float(request.form.get("Classes"))
        Region = float(request.form.get("Region"))

        new_data_scaled = standard_scaler.transform([[
           Temperature,
            RH,
            Ws,
            Rain,
            FFMC,
            DMC,
            ISI,
            Classes,
            Region
        ]])
        result = ridge_model.predict(new_data_scaled)
        return render_template('home.html', result=result[0])
    else:
        return render_template('home.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)