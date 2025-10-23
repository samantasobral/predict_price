from flask import Flask, request, Response
import os
import json
import pickle 
import pandas as pd
import traceback

from empresa.empresa import PredictPrice

model = pickle.load(open('model/final_model.pkl', 'rb'))
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
    
if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run(host='0.0.0.0', port=port, debug=True)