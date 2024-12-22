import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        # x: (batch, seq_length, input_dim)
        # Flatten time and features if needed or process each timestep individually
        # # Combine seq_length and input_dim and Processing
        batch_size, seq_length, input_dim = x.size()
        x = x.reshape(batch_size, -1)  # (batch, seq_length*input_dim)
        
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        decoded = decoded.reshape(batch_size, seq_length, input_dim)
        return decoded

class RNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(RNNEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)

    def forward(self, x):
        # x: (batch, seq_length, input_dim)
        _, (h, c) = self.lstm(x)  
        # h, c: (num_layers, batch, hidden_dim)
        # Use Last hidden state h[-1]
        return h, c

class RNNDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(RNNDecoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, h, c, seq_length):
        # h, c: (num_layers, batch, hidden_dim)
        # Initial input: 0 vector or Learnable Parameter
        # Set Initial Input as 0
        batch_size = h.size(1)
        # Initial token for Decoder (e.g., 0)
        x = torch.zeros(batch_size, 1, self.fc.out_features, device=h.device)
        
        outputs = []
        hidden = (h, c)
        for _ in range(seq_length):
            out, hidden = self.lstm(x, hidden)  # out: (batch, 1, hidden_dim)
            out = self.fc(out)  # (batch, 1, input_dim)
            outputs.append(out)
            x = out  # Unuse teacher-forcing, Reuse previous output
        outputs = torch.cat(outputs, dim=1)  # (batch, seq_length, input_dim)
        return outputs

class RNNAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(RNNAE, self).__init__()
        self.encoder = RNNEncoder(input_dim, hidden_dim, num_layers)
        self.decoder = RNNDecoder(input_dim, hidden_dim, num_layers)

    def forward(self, x):
        # x: (batch, seq_length, input_dim)
        h, c = self.encoder(x)
        seq_length = x.size(1)
        recon = self.decoder(h, c, seq_length)
        return recon

class CNN1DEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, kernel_size=3, stride=2, padding=1):
        super(CNN1DEncoder, self).__init__()
        # input: (batch, input_dim, seq_length)
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, input_dim, seq_length)
        out = self.conv1(x)  # (batch, hidden_dim, seq_length/stride)
        out = self.relu(out)
        return out

class CNN1DDecoder(nn.Module):
    def __init__(self, hidden_dim=64, output_dim=1, kernel_size=3, stride=2, padding=1, output_padding=1):
        super(CNN1DDecoder, self).__init__()
        # Restore sequence length through ConvTranspose1d
        self.deconv1 = nn.ConvTranspose1d(hidden_dim, output_dim, kernel_size=kernel_size, stride=stride,
                                          padding=padding, output_padding=output_padding)
        self.relu = nn.ReLU()
    def forward(self, x):
        # x: (batch, hidden_dim, reduced_seq_length)
        out = self.deconv1(x)  # (batch, output_dim, seq_length)

        return out # raw output

class CNN1DAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, kernel_size=3, stride=2, padding=1, output_padding=1):
        super(CNN1DAE, self).__init__()
        self.encoder = CNN1DEncoder(input_dim, hidden_dim, kernel_size, stride, padding)
        self.decoder = CNN1DDecoder(hidden_dim, input_dim, kernel_size, stride, padding, output_padding)
        
    def forward(self, x):
        # x: (batch, seq_length, input_dim)
        # Apply transpose since CNN input dimension is (batch, channels, length)
        x = x.transpose(1, 2)  # (batch, input_dim, seq_length)
        encoded = self.encoder(x)  # (batch, hidden_dim, reduced_seq_length)
        decoded = self.decoder(encoded)  # (batch, input_dim, seq_length)
        decoded = decoded.transpose(1, 2)  # (batch, seq_length, input_dim)
        return decoded