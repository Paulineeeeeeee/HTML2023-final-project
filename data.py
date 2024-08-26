import numpy as np
import pandas as pd
import os
import json
import torch
import torch.nn as nn
from lstm import LSTMModel
import pickle

holidays = ['2023-10-09', '2023-10-19','2023-11-24']
holidays = pd.to_datetime(holidays)
weather = pd.read_csv('1203.csv')
station = pd.read_csv('station.csv')

def process_json_file(file_path, date_folder, station_file):
    
    # check that the pickle file haven't produced    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # ID         
    records = list(data.values())
    df = pd.DataFrame(records).reindex()
    
    df['station'] = station_file
    df['time'] = pd.to_datetime([f"{date_folder}{time}" for time in data.keys()], format='%Y%m%d%H:%M')
    df['is_weekend'] = df['time'].apply(lambda x: x.weekday() >= 5 or x in holidays).astype(int)
    df['station'] = pd.to_numeric(df['station'])
    df = df.drop(['act','bemp'],axis = 1)
    df.bfill(inplace=True)
    df.ffill(inplace=True)
    df = df[df['time'].dt.minute % 20 == 0]
    df['Date'] =  pd.to_datetime(df['time'].dt.date)
    df['month'] = df['time'].dt.month 
    df['day'] = df['time'].dt.day 
    df['weekday'] = df['time'].dt.weekday 
    df['hour'] = df['time'].dt.hour 
    df['minute'] = df['time'].dt.minute 
    weather['Date'] = pd.to_datetime(weather['Date'])
    df = pd.merge(df, weather, on='Date',how='left')
    df = pd.merge(df, station, on='station',how='left')
    df = df.drop(['Date'], axis=1)
    
    return df

if __name__ == '__main__':
    df_list = []
    for date_folder in  os.listdir('/local/pauline/html.2023.final.data/release'):
        print(f"load {date_folder} file")
        date_folder_path = os.path.join('/local/pauline/html.2023.final.data/release', date_folder)
        if os.path.isdir(date_folder_path):
            for station_file in os.listdir(date_folder_path):
                if station_file.endswith('.json'):
                    file_path = os.path.join(date_folder_path, station_file)
                    station_id = station_file.split('.')[0]

                    data = process_json_file(file_path, date_folder, station_id)
                    df_list.append(data)
                    del data
    
    df = pd.concat(df_list).reset_index(drop=True)
    
    df = df.sort_values(by=['station', 'time'])
    df = df.drop(['time'], axis=1)
    
    # 保存 DataFrame 为 CSV 文件
    df.to_csv('data.csv', index=False)

    print('loading success')
