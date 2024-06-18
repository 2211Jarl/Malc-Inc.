import os

from pymongo import MongoClient

from flask import Flask, request, jsonify, render_template, redirect, url_for, send_from_directory
from flask_cors import CORS

import torch
import torch.nn as nn

import numpy as np

from URLFeatureExtractor import URLFeatureExtractor

app = Flask(__name__, template_folder='UI')
CORS(app)

mongo_uri = 'mongodb+srv://killersaurus161:Jarl_Balgruff2211@malcinc.nfyfpnz.mongodb.net/'
client = MongoClient(mongo_uri)
db = client['thatsPhishy'] 
collection = db['cache'] 

with open("scaler.pkl", "rb") as scaler_file:
   scaler = torch.load(scaler_file, map_location=torch.device('cpu'))

class MLP(nn.Module):
    def __init__(self,dropout=0.4):
        super(MLP,self).__init__()
        self.network=nn.Sequential(
            nn.Linear(in_features=55,out_features=300), 
            nn.ReLU(),
            nn.BatchNorm1d(num_features=300),
            nn.Dropout(p=dropout),

            nn.Linear(in_features=300,out_features=100),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=100),

            nn.Linear(in_features=100,out_features=1),
            nn.Sigmoid()
        )
    def forward(self,x):
        x=self.network(x)
        return x

def getFeaturesOfUrl(url):
    extractor = URLFeatureExtractor(url)
    return np.array(extractor.get_all_features())

model = MLP(dropout=0.4)
model.load_state_dict(torch.load("phishing_model.pkl", map_location=torch.device('cpu')))
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    try:        
        data = request.form if request.form else request.get_json()
        url = data.get('url')

        doc = collection.find_one({"url": url})

        if doc:
            output = doc["prediction"]

        else:
            input_data = getFeaturesOfUrl(url)

            input_data_scaled = scaler.transform(input_data.reshape(1, -1))
            input_tensor = torch.Tensor(input_data_scaled)

            with torch.no_grad():
                output = model(input_tensor).item()

            doc = {"url": url, "prediction": output}
            collection.insert_one(doc)

        # Redirect to the result page with the prediction as a parameter
        return redirect(url_for('result', prediction=output, url=url))
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/')
def index():
    return render_template("templates/index.html")

@app.route('/page1')
def page1():
    return render_template("templates/page1.html", base_template=os.path.join('templates/index.html'))

@app.route('/page2')
def page2():
    return render_template("templates/page2.html", base_template=os.path.join('templates/index.html'))

@app.route('/page3')
def page3():
    return render_template("templates/page3.html", base_template=os.path.join('templates/index.html'))


@app.route('/result')
def result():
    url=request.args.get('url')
    prediction = request.args.get('prediction')
    prediction=float(prediction)
    progress_color = 'green' if prediction < 0.5 else 'red'
    progress_width = prediction * 100
    if prediction < 0.5 : result_text = f'The website {url} is extremely safe to use'
    elif prediction < 0.75: result_text = f'The website is potentially dangerous, please proceed with caution'
    else: result_text = f'The website is unsafe. Please avoid at all costs'
    return render_template("templates/result.html",url=url, prediction=prediction, progress_color=progress_color, progress_width=progress_width, result_text=result_text)

if __name__ == '__main__':

    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)