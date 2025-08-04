# model.py

import numpy as np
from layers import Dense, relu, relu_derivative, sigmoid, sigmoid_derivative
from config import ACTIVATION, LEARNING_RATE, L2_LAMBDA

class NeuralNetwork:
    """
    A simple multi-layer neural network (from scratch).
    """

    def __init__(self, input_dim, hidden_layers, output_dim):
        self.layers = []

        # Input → Hidden layers
        last_dim = input_dim
        for units in hidden_layers:
            self.layers.append(Dense(last_dim, units))
            last_dim = units

        # Hidden → Output
        self.layers.append(Dense(last_dim, output_dim))

    def _activation(self, x):
        """Applies selected activation function."""
        return relu(x) if ACTIVATION == 'relu' else sigmoid(x)

    def _activation_derivative(self, x):
        """Returns derivative of selected activation function."""
        return relu_derivative(x) if ACTIVATION == 'relu' else sigmoid_derivative(x)

    def forward(self, X):
        """
        Forward pass through all layers.
        Stores intermediate activations.
        """
        self.inputs = []  # Raw outputs (Z)
        self.activations = [X]  # Activated outputs (A)

        out = X
        for i, layer in enumerate(self.layers[:-1]):
            z = layer.forward(out)
            out = self._activation(z)

            self.inputs.append(z)
            self.activations.append(out)

        # Final layer (no activation if output is raw logits)
        final_output = self.layers[-1].forward(out)
        self.inputs.append(final_output)
        self.activations.append(final_output)  # Will be activated (sigmoid) in loss

        return final_output

    def backward(self, y_pred, y_true):
        """
        Backward pass (backpropagation) through all layers.
        Applies activation derivatives and updates weights.
        """
        m = y_true.shape[0]  # Batch size

        # Loss derivative for binary cross-entropy with sigmoid
        dz = (y_pred - y_true)  # shape: (batch_size, 1)

        # Output layer (no activation)
        dz = dz  # already dL/dZ since final activation is sigmoid

        # Backprop through layers
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]

            if i != len(self.layers) - 1:
                # Apply activation derivative to hidden layers
                dz = dz * self._activation_derivative(self.inputs[i])

            dz = layer.backward(dz, learning_rate=LEARNING_RATE, l2_lambda=L2_LAMBDA)

