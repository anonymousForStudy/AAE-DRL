import torch
from torch.nn import Linear, ReLU, Module, Sequential, Dropout, BatchNorm1d


class Classifier(Module):
    def __init__(self, input_dim, output_dim):
        super(Classifier, self).__init__()
        seq = [
            Linear(input_dim, 15),
            ReLU(),
            Linear(15, 8),
            ReLU(),
            Linear(8, 15),
            ReLU(),
            Linear(15, 60),
            ReLU(),
            Linear(60, 120),
            ReLU(),
            Dropout(0.5),
            Linear(120, 240),
            ReLU(),
            Linear(240, output_dim),
        ]
        self.seq = Sequential(*seq)


    def forward(self, x):
        x = self.seq(x)
        return x
