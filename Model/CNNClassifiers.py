import torch.nn as nn

class CNN1DClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, num_filters=64, kernel_size=3):
        super(CNN1DClassifier, self).__init__()
        # input_dim: feature dimension
        # Conv1d input: (batch, channels, length)
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=num_filters, kernel_size=kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters, kernel_size=kernel_size, padding=kernel_size//2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_filters, num_classes)
    
    def forward(self, x):
        # x: (batch_size, seq_length, input_dim)
        # Transpose to (batch_size, input_dim, seq_length)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        # AdaptiveAvgPool1d(1): (batch, num_filters, 1)
        x = self.pool(x)
        x = x.squeeze(-1)  # (batch, num_filters)
        x = self.fc(x)  # (batch, num_classes)
        return x

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super(TCNBlock, self).__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                               padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 
                               padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu, self.dropout,
                                 self.conv2, self.chomp2, self.relu, self.dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, num_channels=[64,64], kernel_size=3, dropout=0.2):
        super(TCNClassifier, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            in_ch = input_dim if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]
            dilation_size = 2 ** i
            layers += [TCNBlock(in_ch, out_ch, kernel_size, dilation=dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x):
        # x: (batch_size, seq_length, input_dim)
        # Transpose to (batch_size, input_dim, seq_length)
        x = x.transpose(1, 2)
        y = self.network(x)  # (batch, out_ch, seq_length)
        # Global average pooling
        y = y.mean(dim=2)  # (batch, out_ch)
        y = self.fc(y)  # (batch, num_classes)
        return y