import torch
import torch.nn as nn


class Self_Attn(nn.Module):
    """ Self-Attention Layer for Tabular Data """

    def __init__(self, in_dim, activation='relu'):
        """
        Initialize self-attention layer for tabular data.

        Parameters
        ----------
        in_dim : int
            Number of features in the input data.
        activation : str
            Activation type (e.g., 'relu').
        """
        super(Self_Attn, self).__init__()
        self.activation = activation

        # Linear layers to learn key, query, and value representations
        self.query_layer = nn.Linear(in_dim, in_dim // 8)
        self.key_layer = nn.Linear(in_dim, in_dim // 8)
        self.value_layer = nn.Linear(in_dim, in_dim)

        # Learnable scaling parameter
        self.gamma = nn.Parameter(torch.zeros(1))

        # Softmax for attention weights
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Forward pass of self-attention for tabular data.

        Parameters
        ----------
        x : torch.Tensor
            Input data (shape: B x F, where F is the number of features).

        Returns
        -------
        out : torch.Tensor
            Self-attention value + input data (shape: B x F).
        attention : torch.Tensor
            Attention coefficients (shape: B x F x F).
        """

        # Compute query, key, and value projections
        query = self.query_layer(x)  # shape: B x (F // 8)
        key = self.key_layer(x)  # shape: B x (F // 8)
        value = self.value_layer(x)  # shape: B x F

        # Compute attention scores
        energy = torch.matmul(query.squeeze(1), key.squeeze(1).T)  # shape: B x F x F
        attention = self.softmax(energy)  # shape: B x F x F

        # Compute self-attention output
        out = torch.matmul(attention, value)  # shape: B x F
        out = self.gamma * out + x  # Residual connection

        return out, attention



class Generator(nn.Module):
    """Generator for Tabular Data."""

    def __init__(self, out_size=30, z_dim=10, hidden_dim=64):
        """
        Initialize the Generator model for tabular data.

        Parameters
        ----------
        out_size : int
            Number of features in the generated output (size of tabular data).
        z_dim : int
            Size of the latent vector.
        hidden_dim : int
            Size of hidden layers.
        """
        super(Generator, self).__init__()
        self.out_size = out_size

        # Layer 1: Latent vector to hidden representation
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.act1 = nn.ReLU()

        # Layer 2: Hidden to smaller hidden
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.act2 = nn.ReLU()

        # Self-attention layer for tabular data
        self.attn = Self_Attn(hidden_dim // 2, activation='relu')

        # Final layer: Hidden to output
        self.fc_out = nn.Linear(hidden_dim // 2, out_size)
        self.act_out = nn.Tanh()

    def forward(self, z):
        """
        Generate data from the latent vector.

        Parameters
        ----------
        z : torch.Tensor
            Latent vector input (shape: B x z_dim).

        Returns
        -------
        out : torch.Tensor
            Generated data (shape: B x out_size).
        p1 : torch.Tensor
            Attention coefficients (shape: B x F x F).
        """
        # Layer 1
        out = self.fc1(z)  # B x z_dim ---> B x hidden_dim
        out = self.bn1(out)  # Normalize
        out = self.act1(out)  # Activation

        # Layer 2
        out = self.fc2(out)  # B x hidden_dim ---> B x (hidden_dim // 2)
        out = self.bn2(out)  # Normalize
        out = self.act2(out)  # Activation

        # Attention
        out, p1 = self.attn(out)  # Apply self-attention

        # Final output
        out = self.fc_out(out)  # B x (hidden_dim // 2) ---> B x out_size
        out = self.act_out(out)  # Activation (e.g., Tanh)

        return out, p1


class Discriminator(nn.Module):
    """Discriminator for Tabular Data."""

    def __init__(self, input_size=30, hidden_dim=64):
        """
        Initialize Discriminator model for tabular data.

        Parameters
        ----------
        input_size : int
            Number of features in the input data.
        hidden_dim : int
            Size of the hidden layers.
        """
        super(Discriminator, self).__init__()
        self.input_size = input_size

        # Layer 1: Input to hidden representation
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.act1 = nn.LeakyReLU(0.1)

        # Layer 2: Hidden to smaller hidden
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.act2 = nn.LeakyReLU(0.1)

        # Self-attention layer for tabular data
        self.attn = Self_Attn(hidden_dim // 2, activation='relu')

        # Final layer: Hidden to a single output (binary classification)
        self.fc_out = nn.Linear(hidden_dim // 2, 1)
        self.act_out = nn.Sigmoid()

    def forward(self, x):
        """
        Perform forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input of size B x input_size.

        Returns
        -------
        out : torch.Tensor
            Output of size B x 1 (classification score).
        p1 : torch.Tensor
            Attention coefficients (shape: B x F x F).
        """
        # Layer 1
        out = self.fc1(x)  # B x input_size ---> B x hidden_dim
        out = self.act1(out)

        # Layer 2
        out = self.fc2(out)  # B x hidden_dim ---> B x (hidden_dim // 2)
        out = self.act2(out)

        # Attention
        out, p1 = self.attn(out)  # Apply self-attention

        # Final output
        out = self.fc_out(out)  # B x (hidden_dim // 2) ---> B x 1
        out = self.act_out(out)

        return out, p1  # Output and attention coefficients
