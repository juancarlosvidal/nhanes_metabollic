# Neural network model definitions for binary classification and regression.
import torch
import torch.nn as nn
from utils import get_device


def init_weights(m):
    """Initialize Linear layer weights with Xavier uniform and biases with a small constant."""
    if isinstance(m, nn.Linear):
        # Opt 1
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
        # Opt 2
        # nn.init.orthogonal_(m.weight)
        # nn.init.constant_(m.bias, 0)
        # Opt 3
        # nn.init.uniform_(m.weight)


class LogisticRegression(nn.Module):
    """Single-layer logistic regression model (linear + sigmoid)."""

    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.linear.apply(init_weights)

    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs


class BinaryClassificationModel(nn.Module):
    """
    Multilayer Perceptron for binary classification.
    Each hidden layer consists of: Linear -> LayerNorm -> ReLU -> Dropout.
    The output layer uses a Sigmoid activation.
    """

    def __init__(self, input_size, output_size, dropout, hidden_sizes):
        super(BinaryClassificationModel, self).__init__()
        if len(hidden_sizes) <= 0:
            raise Exception("Sorry, the number of hidden layers must be greater than 0")
        self.layers = nn.ModuleList()
        self.input_size = input_size
        # Build hidden layers dynamically from the provided size list
        for size in hidden_sizes:
            self.layers.append(nn.Linear(input_size, size))
            # self.layers.append(nn.BatchNorm1d(size))
            self.layers.append(nn.LayerNorm(size))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(p=dropout))
            input_size = size  # For the next layer
        self.layers.append(nn.Linear(input_size, output_size))
        self.layers.append(nn.Sigmoid())
        self.device = get_device()
        self.to(self.device)
        self.layers.apply(init_weights)

    def forward(self, input_data):
        """Forward pass: sequentially apply all layers."""
        for layer in self.layers:
            input_data = layer(input_data)
        return input_data
    #
    #
    #
    #
    #
    #     """Multilayer Perceptron for classification"""
    #     super(BinaryClassificationModel, self).__init__()
    #     self.layers = nn.Sequential(
    #         h1 = input_dim
    #         for h2 in hidden_dims:
    #             nn.Linear(h1, h2),
    #             nn.BatchNorm1d(hidden_dim * 4),
    #             nn.ReLU(),
    #             nn.Dropout(p=0.20),
    #
    #         nn.Linear(hidden_dim * 4, hidden_dim * 3),
    #         # nn.LeakyReLU(),
    #         nn.BatchNorm1d(hidden_dim * 3),
    #         nn.ReLU(),
    #         nn.Dropout(p=0.50),
    #         nn.Linear(hidden_dim * 3, hidden_dim * 2),
    #         # nn.LeakyReLU(),
    #         nn.BatchNorm1d(hidden_dim * 2),
    #         nn.ReLU(),
    #         nn.Dropout(p=0.50),
    #         nn.Linear(hidden_dim * 2, hidden_dim * 1),
    #         # nn.LeakyReLU(),
    #         nn.BatchNorm1d(hidden_dim * 1),
    #         nn.ReLU(),
    #         nn.Dropout(p=0.50),
    #         nn.Linear(hidden_dim * 1, int(hidden_dim/2)),
    #         nn.BatchNorm1d(int(hidden_dim/2)),
    #         nn.ReLU(),
    #         nn.Dropout(p=0.50),
    #         nn.Linear(int(hidden_dim/2), output_dim),
    #         nn.Sigmoid()
    #     )
    #     self.layers.apply(init_weights)
    #
    # def forward(self, x):
    #     """Forward pass"""
    #     return self.layers(x)


# class RegressionModel(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         """Multilayer Perceptron for conformal inference"""
#         super(RegressionModel, self).__init__()
#         self.layers = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             # nn.LeakyReLU(),
#             nn.ReLU(),
#             # nn.BatchNorm1d(64),
#             nn.Dropout(p=0.50),
#             nn.Linear(hidden_dim, output_dim)
#         )
#         self.layers.apply(init_weights)
#
#     def forward(self, x):
#         """Forward pass"""
#         return self.layers(x)

class RegressionModel(nn.Module):
    """
    Multilayer Perceptron for regression (quantile estimation in conformal inference).
    Each hidden layer consists of: Linear -> BatchNorm1d -> ReLU -> Dropout.
    No activation on the output layer.
    """

    def __init__(self, input_size, output_size, dropout, hidden_sizes):
        super(RegressionModel, self).__init__()
        if len(hidden_sizes) <= 0:
            raise Exception("Sorry, the number of hidden layers must be greater than 0")
        self.layers = nn.ModuleList()
        self.input_size = input_size
        # Build hidden layers dynamically from the provided size list
        for size in hidden_sizes:
            self.layers.append(nn.Linear(input_size, size))
            # self.layers.append(nn.BatchNorm1d(size))
            self.layers.append(nn.LayerNorm(size))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(p=dropout))
            input_size = size  # For the next layer
        self.layers.append(nn.Linear(input_size, output_size))
        self.device = get_device()
        self.to(self.device)
        self.layers.apply(init_weights)

    def forward(self, input_data):
        """Forward pass: sequentially apply all layers."""
        for layer in self.layers:
            input_data = layer(input_data)
        return input_data
