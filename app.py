from flask import Flask, request, Response
import os
import json
import pickle 
import pandas as pd
import traceback
import requests

from empresa.empresa import PredictPrice

import os
import requests
import pickle

def carregar_modelo_grande(file_id, destino='model.pkl'):
    if not os.path.exists(destino):
        # Link direto para download via navegador
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        response = requests.get(url, allow_redirects=True)

        if response.status_code == 200:
            with open(destino, 'wb') as f:
                f.write(response.content)
        else:
            raise Exception(f"Erro ao baixar modelo: {response.status_code}")

    with open(destino, 'rb') as f:
        return pickle.load(f)

# Chamada do carregamento
model = carregar_modelo_grande("19PunwujGRBa2f9QGWx_GqjZ6ruPExEbr")

app = Flask(__name__)

@app.route('/empresa/predict', methods=['POST'])
def price_predict():
    try:
        test_json = request.get_json()

        if test_json:
            if isinstance(test_json, dict):
                test_raw = pd.DataFrame(test_json, index=[0])
            else:
                test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())

            pipeline = PredictPrice()

            df_format = pipeline.data_formatation(test_raw)
            df_feature = pipeline.feature_engineering(df_format)
            df_preparation = pipeline.data_preparation(df_feature)
            df_predict = pipeline.get_predictions(model, df_preparation, df_format)

            return Response(df_predict, status=200, mimetype='application/json')
        else:
            return Response('{}', status=200, mimetype='application/json')

    except Exception as e:
        print("❌ Erro interno na API:")
        traceback.print_exc()  # mostra o erro completo no terminal
        return Response(json.dumps({'erro': str(e)}), status=500, mimetype='application/json')


@app.route('/ping', methods=['GET'])
def ping():
    return "pong", 200
    





