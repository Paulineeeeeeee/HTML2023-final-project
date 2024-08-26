import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lstm import LSTMModel
from lstm import GRUModel
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold


def weighted_absolute_error(y_pred, y_true, s_i):
    
    total_loss = 0
    total_data = len(y_true)
    s_i = torch.clamp(s_i, min=1e-8)

    for i in range(0, total_data, 1000):
        # 提取当前批次的数据
        y_pred_batch = y_pred[i : min(i + 1000, total_data)]
        y_true_batch = y_true[i : min(i + 1000, total_data)]
        s_i_batch = s_i[i : min(i + 1000, total_data)]

        # 计算绝对误差
        abs_error = torch.abs(y_true_batch - y_pred_batch) / s_i_batch

        # 计算权重
        weights = (torch.abs(y_true_batch / s_i_batch - 1/3) + torch.abs(y_true_batch / s_i_batch - 2/3))

        # 计算加权误差
        weighted_error = weights * abs_error * 3

        # 累加损失的平均值或总和
        total_loss += torch.sum(weighted_error) 


    # 计算所有批次的平均损失
    avg_loss = total_loss / total_data

    return avg_loss


if __name__ == '__main__':

    torch.cuda.empty_cache()

    data = pd.read_csv("data.csv")
    X = np.array(data.drop(['sbi'],axis = 1)).astype('float32')
    y = np.array(data['sbi']).astype('float32')
    si = np.array(data['tot']).astype('float32')

    # prediction
    print("prediction")

    with open('private_id.pkl', 'rb') as file:
        private_id = pickle.load(file)
    with open('private.pkl', 'rb') as file:
        private = pickle.load(file)

    private = private[['tot','station','is_weekend','month','day','weekday','hour','minute','high','low','weather','lng','lat']]
    
    print('min max scaler')
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(X)
    private = scaler.transform(private)
    private = torch.from_numpy(np.array(private)).float()
    
    #preprocessing

    kf = KFold(n_splits=5)
    for train_index, test_index in kf.split(X):
        X_train, X_val = X[train_index], X[test_index]
        y_train, y_val = y[train_index], y[test_index]
        si_train, si_val = si[train_index], si[test_index]
        
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)

            # torch
        X_train = torch.from_numpy(X_train).float()
        X_val = torch.from_numpy(X_val).float()
        y_train = torch.from_numpy(y_train).float()
        y_val = torch.from_numpy(y_val).float()
        si_train = torch.from_numpy(si_train).float()
        si_val = torch.from_numpy(si_val).float()
        


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'device : {device}')

        print("model struct")
        input_dim = len(data.columns) - 1    # 输入特征维度
        hidden_dim = 64      # 隐藏层维度
        layer_dim = 2    # LSTM层的数量
        output_dim = 1    # 输出维度
        patience = 30
        patience_counter = 0
        best_val_loss = float('inf')

        # model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim).to(device)
        model = GRUModel(input_dim, hidden_dim, layer_dim, output_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        iter=0
        print("start training")
        for epoch in range(100):  
            model.train()
            total_data = len(X_train)
            total_loss = 0
            for i in range(0 , total_data , 500 ):
                train_X_tensor = X_train[i:min(i + 500, total_data)].to(device)
                outputs_train = model(train_X_tensor).to('cpu')
                loss = weighted_absolute_error(outputs_train, y_train[i:min(i + 500, total_data)] , si_train[i:min(i + 500, total_data)])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(train_X_tensor)
            
            average_loss = total_loss / total_data
            scheduler.step()
            print(f"Average Loss: {average_loss}")
                        
            # 验证集上的前向传播
            model.eval() 
            with torch.no_grad():  # 确保在评估模型时不计算梯度
                X_val = X_val.to(device)
                outputs_val = model(X_val).to('cpu').view(-1)
                val_loss = weighted_absolute_error(outputs_val, y_val, si_val)
                

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0  # 重置耐心计数器
            else:
                patience_counter += 1  # 增加耐心计数器
            
            print("Epoch {:05d} | Loss {:.4f}".format(epoch, val_loss.item()))

            if patience_counter >= patience:
                print("Stopping early due to no improvement in validation loss.")
                break


        model.eval() 
        with torch.no_grad():
            private = private.to(device)
            outputs_private = model(private).detach().cpu().numpy().flatten()
        

        results_df = pd.DataFrame({
            'id': private_id,
            'sbi': outputs_private
        })

        results_df.to_csv(f'outputs/private_predicted_gru{best_val_loss}.csv', index=False)

