import torch 
import torch.nn as nn

class RNN(nn.Module):
    """A simple Recurrent Neural Network (RNN) model."""

    def __init__(self, input_size, hidden_size, output_size, num_layers=1, rnn_type="simple", nonlinearity="tanh"):
        super(RNN, self).__init__()
        
        if rnn_type.lower() == "simple":
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, nonlinearity=nonlinearity)
        elif rnn_type.lower() == "lstm":
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        elif rnn_type.lower() == "gru":
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        else:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}")

        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, h=None):
        rnn_out, h = self.rnn(x, h)
        out = self.fc(rnn_out)  # Take the output of the last time step
        return out, h