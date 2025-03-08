from torch.nn import Linear, ReLU, Module, Sequential, BatchNorm1d, ModuleDict, Softmax, Tanh, MSELoss, \
    CrossEntropyLoss, Sigmoid, BCELoss


class Encoder(Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.seq = Sequential(
            Linear(30, 30),
            ReLU(),
            BatchNorm1d(30),
            Linear(30, 15),
            ReLU(),
            BatchNorm1d(15),
            Linear(15, 30),
            ReLU(),
            BatchNorm1d(30),
            Linear(30, 60),
            ReLU(),
            BatchNorm1d(60),
            Linear(60, 120),
            ReLU(),
            BatchNorm1d(120),
            # bottleneck
            Linear(120, 2),
            ReLU(),
        )

    def forward(self, x):
        x = self.seq(x)
        return x

class Decoder(Module):
    def __init__(self, discrete, continuous, binary):
        super(Decoder, self).__init__()
        self.discrete_features = discrete
        self.continuous_features = continuous
        self.binary_features = binary
        self.seq = Sequential(
            Linear(2, 60),
            ReLU(),
            BatchNorm1d(60),
            Linear(60, 30),
            ReLU(),
            BatchNorm1d(30),
            Linear(30, 15),
            ReLU(),
            BatchNorm1d(15),
            Linear(15, 30),
            ReLU(),
        )
        self.discrete_out = {feature: Linear(30, num_classes)
                             for feature, num_classes in discrete.items()}
        self.continuous_out = {feature: Linear(30, 1)
                               for feature in continuous}
        self.binary_out = {feature: Linear(30, 2)
                               for feature in binary}

        self.discrete_out = ModuleDict(self.discrete_out)
        self.continuous_out = ModuleDict(self.continuous_out)
        self.binary_out = ModuleDict(self.binary_out)


        self.softmax = Softmax()
        self.tanh = ReLU()
        self.sigmoid = Sigmoid()
        self.mse = MSELoss()
        self.ce = CrossEntropyLoss()
        self.bce = BCELoss()


    def forward(self, x):
        shared_features = self.seq(x)
        discrete_outputs = {}
        continuous_outputs = {}
        binary_outputs = {}

        for feature in self.discrete_features:
            logits = self.discrete_out[feature](shared_features)
            discrete_outputs[feature] = self.softmax(logits)

        for feature in self.continuous_features:
            continuous_outputs[feature] = self.tanh(self.continuous_out[feature](shared_features))

        for feature in self.binary_features:
            binary_outputs[feature] = self.sigmoid(self.binary_out[feature](shared_features))

        return discrete_outputs, continuous_outputs, binary_outputs

    def compute_loss(self, outputs, targets):
        discrete_outputs, continuous_outputs, binary_outputs = outputs
        discrete_targets, continuous_targets, binary_targets = targets
        total_loss = 0

        # For discrete features, check against discrete_targets dictionary
        for feature in self.discrete_features:
            if feature in discrete_targets:  # Changed from targets to discrete_targets
                total_loss += self.ce(discrete_outputs[feature], discrete_targets[feature])

        # For continuous features, check against continuous_targets dictionary
        for feature in self.continuous_features:
            if feature in continuous_targets:  # Changed from targets to continuous_targets
                total_loss += self.mse(continuous_outputs[feature], continuous_targets[feature])

        for feature in self.binary_features:
            if feature in binary_targets:  # Changed from targets to continuous_targets
                total_loss += self.bce(binary_outputs[feature], binary_targets[feature])

        return total_loss
