# libraries 
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence


class Encoder(nn.Module):
    """Encoder Class"""
    def __init__(self, im_chan=30, output_chan=10, hidden_dim=16):
        """
        initialize Encoder model

        Parameters
        ----------
        im_chan : int
            channel number of input image
        output_chan : int
            channel number of encoded output (latent space)
        hidden_dim : int
            channel number of output of first hidden block layer (a measure of model capacity)
        """
        super(Encoder, self).__init__()
        self.z_dim = output_chan
        self.disc = nn.Sequential(
            self.make_disc_block(im_chan, hidden_dim),
            self.make_disc_block(hidden_dim, hidden_dim * 2),
            self.make_disc_block(hidden_dim * 2, output_chan, final_layer=True),
        )

    def make_disc_block(self, input_channels, output_channels, final_layer=False):
        """
        Function to return a sequence of operations corresponding to a encoder block of the VAE, 
        corresponding to a convolution, a batchnorm (except for in the last layer), and an activation
        
        Parameters
        ----------
        input_channels : int
            how many channels the input feature representation has
        output_channels : int 
            how many channels the output feature representation should have
        kernel_size : int 
            the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
        stride : int 
            the stride of the convolution
        final_layer : bool 
            whether we're on the final layer (affects activation and batchnorm)
        """       
        if not final_layer:
            return nn.Sequential(
                # nn.Conv1d(input_channels, output_channels, kernel_size, stride),
                nn.Linear(input_channels, output_channels),
                nn.LayerNorm(output_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            return nn.Sequential(
                # nn.Conv1d(input_channels, output_channels, kernel_size, stride),
                nn.Linear(input_channels, output_channels)
            )

    def forward(self, image):
        """
        Function for completing a forward pass of the Encoder: Given an image tensor, 
        returns a 1-dimension tensor representing fake/real.

        Parameters
        ----------
        image: torch.Tensor
            a flattened image tensor with dimension (im_dim)

        Returns
        -------
        torch.Tensor
            encoded result (latent space)

        """
        encoding = self.disc(image)
        return encoding


class Decoder(nn.Module):
    """Decoder Class"""
    def __init__(self, z_dim, im_chan, hidden_dim, discrete, continuous, binary):
        super(Decoder, self).__init__()
        self.z_dim = z_dim
        self.discrete_features = discrete
        self.continuous_features = continuous
        self.binary_features = binary
        self.gen = nn.Sequential(
            nn.Linear(z_dim, hidden_dim*4),
            nn.LayerNorm(hidden_dim*4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim*4, hidden_dim*2),
            nn.LayerNorm(hidden_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, im_chan)
        )

        self.discrete_out = {feature: nn.Linear(im_chan, num_classes)
                             for feature, num_classes in discrete.items()}
        self.continuous_out = {feature: nn.Linear(im_chan, 1)
                               for feature in continuous}
        self.binary_out = {feature: nn.Linear(im_chan, 2)
                               for feature in binary}

        self.discrete_out = nn.ModuleDict(self.discrete_out)
        self.continuous_out = nn.ModuleDict(self.continuous_out)
        self.binary_out = nn.ModuleDict(self.binary_out)

        self.softmax = nn.Softmax()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCELoss()

    def forward(self, x):
        shared_features = self.gen(x)
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

def kl_divergence_loss(q_dist):
    """
    to calculate kl divergence distance for loss
    """
    return kl_divergence(
        q_dist, Normal(torch.zeros_like(q_dist.mean), torch.ones_like(q_dist.stddev))
    ).sum(-1)

