import torch
import torch.nn as nn

# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim = 1):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        # self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, bidirectional=True)
        # self.fc = nn.Linear(hidden_dim, output_dim)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        x = x.unsqueeze(1)
        
        # h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device='cuda')
        # c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device='cuda')

        h0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim).requires_grad_().to(device='cuda')
        c0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim).requires_grad_().to(device='cuda')
        
        # 从LSTM层获取隐藏状态
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        out = self.dropout(out)
        
        # 选择最后一个时间点的隐藏状态
        # out = self.fc(out[:, -1, :]) # lstm
        out = self.fc(torch.cat((hn[-2,:,:], hn[-1,:,:]), dim = 1)) # bilstm
        return out

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim = 1):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.gru = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        x = x.unsqueeze(1)
        
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device='cuda')
        # c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device='cuda')
        
        # 从LSTM层获取隐藏状态
        out, (hn) = self.gru(x, (h0.detach()))

        out = self.dropout(out)
        
        # 选择最后一个时间点的隐藏状态
        out = self.fc(out[:, -1, :]) # lstm
        return out