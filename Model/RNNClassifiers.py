import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        # x: (batch_size, seq_length, input_dim)
        out, (hn, cn) = self.lstm(x)
        # hn: (num_layers, batch_size, hidden_dim)
        # Take the last hidden state of the LSTM
        hn = hn[-1]  # (batch_size, hidden_dim)
        out = self.fc(hn)  # (batch_size, num_classes)
        return out

class GRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(GRUClassifier, self).__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        # x: (batch_size, seq_length, input_dim)
        out, hn = self.gru(x)
        # hn: (num_layers, batch_size, hidden_dim)
        hn = hn[-1]  # (batch_size, hidden_dim)
        out = self.fc(hn)  # (batch_size, num_classes)
        return out