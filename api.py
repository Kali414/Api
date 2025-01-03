from flask import Flask,url_for,jsonify,request
import requests
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
app=Flask(__name__)

model=joblib.load("model.joblib")
target=list(np.load("target.npy"))
@app.route('/')
@app.route('/predict',methods=["POST"])
def predict():
    try:
        data=request.get_json()
        if not data:
            return jsonify({"error":"No input data provided"}),400
        
        else:
            data=pd.DataFrame([{ 
                "sepal length (cm)": data["sepal length (cm)"],
                "sepal width (cm)": data["sepal width (cm)"],
                "petal length (cm)": data["petal length (cm)"],
                "petal width (cm)": data["petal width (cm)"]
                }])
            scaled_data=StandardScaler().fit_transform(data)
            prediction=model.predict(data)

            return jsonify({"Prediction":target[prediction[0]]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__=="__main__":
    app.run(debug=True,port=6900)