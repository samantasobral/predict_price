import pandas as pd
import pickle
import numpy as np
import json

import os
import pickle

class PredictPrice(object):
    def __init__(self):
        # Caminho absoluto para a pasta raiz do projeto
        ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        PARAM_DIR = os.path.join(ROOT_DIR, 'parameter')

        def carregar(nome_arquivo):
            caminho = os.path.join(PARAM_DIR, nome_arquivo)
            print(f"🔍 Carregando: {caminho}")  # debug opcional
            return pickle.load(open(caminho, 'rb'))

        # Carregando os encoders e scalers
        self.cia_aerea_encoder = carregar('cia_aerea_label_encoder.pkl')
        self.cidade_origem_encoder = carregar('cidade_origem_label_encoder.pkl')
        self.escalas_scaler = carregar('escalas_scaler.pkl')
        self.cidade_destino_encoder = carregar('cidade_destino_label_encoder.pkl')
        self.dias_restantes_scaler = carregar('dias_restantes_scaler.pkl')
        self.cd_voo_encoder = carregar('cd_voo_label_encoder.pkl')
        self.duracao_horas_scaler = carregar('duracao_horas_scaler.pkl')
        self.periodo_chegada_encoder = carregar('periodo_chegada_label_encoder.pkl')
        self.day_of_week_scaler = carregar('day_of_week_scaler.pkl')
        self.day_scaler = carregar('day_scaler.pkl')
        self.year_week_encoder = carregar('year_week_label_encoder.pkl')
    
    def classificar_periodo(self, hora):
        if pd.isna(hora):
            return 'desconhecido'
        hora_int = hora.hour
        if 5 <= hora_int < 12:
            return 'manhã'
        elif 12 <= hora_int < 18:
            return 'tarde'
        elif 18 <= hora_int < 24:
            return 'noite'
        else:
            return 'madrugada'
    
    def data_formatation(self, df):
        df.rename(columns={'Unnamed: 0': 'desconhecido', 'date': 'data_voo', 'airline': 'cia_aerea', 'ch_code': 'ch_code', 'num_code': 'num_code', 'dep_time': 'hora_partida','from': 'cidade_origem', 'time_taken': 'duracao', 'stop': 'escalas', 'arr_time': 'hora_chegada', 'to': 'cidade_destino', 'price': 'preco', 'days_left': 'dias_restantes'}, inplace=True)
        #criando a coluna de código do vôo:
        df['cd_voo'] = df['ch_code'] + df['num_code'].astype(str)

        #ajustando o tipo de dados:
        df['preco'] = df['preco'].astype(str).str.replace(',', '', regex=False).astype(int)
        df['dias_restantes'] = df['dias_restantes'].astype('int64')
        df['data_voo'] = pd.to_datetime(df['data_voo'], format='%d-%m-%Y')
        df['hora_partida'] = pd.to_datetime(df['hora_partida'], format='%H:%M').dt.time
        df['hora_chegada'] = pd.to_datetime(df['hora_chegada'], format='%H:%M').dt.time

        #Tempo de duração do vôo
        df['duracao_original'] = df['duracao']  # salva a duração original
        df['duracao_convertida'] = pd.to_timedelta(df['duracao'].str.replace('h', 'hours').str.replace('m', 'minutes'),errors='coerce')
        
        #nova coluna, com duração em horas:
        df['duracao_horas'] = df['duracao_convertida'].dt.total_seconds() / 3600

        #Ajustando os nulos:
        df['duracao_horas'] = df['duracao_horas'].fillna(1)

        #Escalas:
        df['escalas'] = df['escalas'].str[0]
        df['escalas'] = df['escalas'].replace('n', '0')
        df['escalas'] = df['escalas'].astype('int64')

        #horário de saída e de chegada:
        df['periodo_partida'] = df['hora_partida'].apply(self.classificar_periodo)
        df['periodo_chegada'] = df['hora_chegada'].apply(self.classificar_periodo)
        
        #excluindo as colunas:
        df.drop(columns = ['ch_code', 'num_code', 'duracao', 'duracao_original', 'duracao_convertida', 'hora_partida', 'hora_chegada', 'desconhecido'], inplace = True)
        return df
    
       
    def feature_engineering(self, df2):
        #year
        df2['year'] = df2['data_voo'].dt.year
        #month
        df2['month'] = df2['data_voo'].dt.month
        #day
        df2['day'] = df2['data_voo'].dt.day
        #year week
        df2['year_week'] = df2['data_voo'].dt.strftime('%Y-%W')
        #dia da semana
        df2['day_of_week'] = df2.data_voo.dt.day_of_week.astype(int)
        #fim de semana
        df2['is_weekend'] = np.where(df2['day_of_week'].isin([5,6]),1,0)
        df2.drop(columns = ['year', 'data_voo'], inplace = True)
        return df2
    
    def data_preparation(self, df3):
        df3 = df3[['cia_aerea', 'cidade_origem', 'escalas', 'cidade_destino', 'dias_restantes','cd_voo', 'duracao_horas','periodo_chegada', 'day', 'year_week', 'day_of_week']]
        df3['cia_aerea'] = self.cia_aerea_encoder.transform(df3[['cia_aerea']])
        df3['cidade_origem'] = self.cidade_origem_encoder.transform(df3[['cidade_origem']])
        df3['escalas'] = self.escalas_scaler.transform(df3[['escalas']])
        df3['cidade_destino'] = self.cidade_destino_encoder.transform(df3[['cidade_destino']])
        df3['dias_restantes'] = self.dias_restantes_scaler.transform(df3[['dias_restantes']])
        df3['cd_voo'] = self.cd_voo_encoder.transform(df3[['cd_voo']])
        df3['duracao_horas'] = self.duracao_horas_scaler.transform(df3[['duracao_horas']])
        df3['periodo_chegada'] = self.periodo_chegada_encoder.transform(df3[['periodo_chegada']])
        df3['day'] = self.day_scaler.transform(df3[['day']])
        df3['year_week'] = self.year_week_encoder.transform(df3[['year_week']])
        df3['day_of_week']= self.day_of_week_scaler.transform(df3[['day_of_week']])
        return df3
    
    def get_predictions(self, model, test_data, original_data):
        pred = model.predict(test_data)
        pred = np.array(pred)
        original_data['prediction'] = np.expm1(pred)
        return original_data.to_json(orient='records')