import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
import numpy as np

weather = pd.read_csv('1203.csv')
station = pd.read_csv('station.csv')
data = pd.read_csv('data.csv')

def prediction_data(date_range):
    id = np.loadtxt('sno_test_set.txt').astype(int)
    
    X_prediction = []

    for station_id in id:
        for single_date in date_range:
            features = {
                'station': station_id,
                'time': pd.to_datetime(single_date)
                # 如果有其他相关特征，也可以在这里添加
            }
            X_prediction.append(features)
    df = pd.DataFrame(X_prediction)
    df['station'] = pd.to_numeric(df['station'],errors = 'coerce')
    df['is_weekend'] = df['time'].apply(lambda x: x.weekday() >= 5).astype(int)
    df['Date'] =  pd.to_datetime(df['time'].dt.date)
    df['month'] = df['time'].dt.month 
    df['day'] = df['time'].dt.day 
    df['weekday'] = df['time'].dt.weekday 
    df['minute'] = df['time'].dt.minute 
    df['hour'] = df['time'].dt.hour 
    weather['Date'] = pd.to_datetime(weather['Date'])
    df = pd.merge(df, weather, on='Date',how='left')
    df = pd.merge(df, station, on='station',how='left')
    df = df.drop(['Date'], axis=1)
    df = df.drop(['time'], axis=1)
    return df


if __name__ == '__main__':
    

    data = data[['station' , 'tot']]
    
    station_max_tot = data.groupby('station')['tot'].max()

    # 将每个 station 的 tot 更新为平均值
    data['tot'] = data['station'].map(station_max_tot)

    unique_totals_per_station = data.groupby('station')['tot'].nunique()

    # 检查是否每个 station 都只有一个唯一的 tot 值
    all_stations_have_one_total = (unique_totals_per_station == 1).all()


    stations_with_multiple_totals = unique_totals_per_station[unique_totals_per_station > 1]

    data = data.drop_duplicates()

    date_range_1 = pd.date_range(start='2023-10-21 00:00', end='2023-10-24 23:40', freq='20T')
    date_range_2 = pd.date_range(start='2023-12-04 00:00', end='2023-12-10 23:40', freq='20T')
    # date_range_2 = pd.date_range(start='2023-12-11 00:00', end='2023-12-17 23:40', freq='20T')
    # date_range_2 = pd.date_range(start='2023-12-18 00:00', end='2023-12-24 23:40', freq='20T')
    public = prediction_data(date_range_1)
    private = prediction_data(date_range_2)
    public = public.merge(data[['station','tot']], on='station', how='left')
    private = private.merge(data[['station','tot']], on='station', how='left')
        # output id 20231204_500101001_11:20
    public_id = ('2023' + public['month'].astype(str) + public['day'].astype(str).str.zfill(2) +
                '_' + public['station'].astype(str) + "_" + public['hour'].astype(str).str.zfill(2) + ':' + public['minute'].astype(str).str.zfill(2))
    private_id = ('2023' + private['month'].astype(str) + private['day'].astype(str).str.zfill(2) +
                '_' + private['station'].astype(str) + "_" + private['hour'].astype(str).str.zfill(2) + ':' + private['minute'].astype(str).str.zfill(2))

    public_id = public_id.drop_duplicates()
    private_id = private_id.drop_duplicates()
    
    private_id = pd.concat([public_id, private_id], ignore_index=True)
    private = pd.concat([public, private], ignore_index=True)
    
    with open('private_id.pkl', 'wb') as file:
        pickle.dump(private_id, file)

    with open('private.pkl', 'wb') as file:
        pickle.dump(private, file)

    private.to_csv('private.csv', index=False)
    
    print("12/03 prediction file success")

