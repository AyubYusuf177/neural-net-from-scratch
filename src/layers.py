# layers.py

import numpy as np

# Activation Functions
def relu(x):
    """Applies the ReLU activation function: max(0, x)."""
    return np.maximum(0, x)

def relu_derivative(x):
    """Derivative of ReLU: 1 where x > 0, else 0."""
    return (x > 0).astype(float)

def sigmoid(x):
    """Applies the sigmoid function: 1 / (1 + e^(-x))"""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Derivative of sigmoid: sigmoid(x) * (1 - sigmoid(x))"""
    s = sigmoid(x)
    return s * (1 - s)

class Dense:
    """
    Fully connected (dense) layer.
    """

    def __init__(self, input_size, output_size):
        # Initialize weights with small random numbers
        self.W = np.random.randn(input_size, output_size) * 0.01
        self.b = np.zeros((1, output_size))

        # Placeholders for inputs/outputs for backprop
        self.input = None
        self.output = None

    def forward(self, X):
        """Performs forward pass through this layer."""
        self.input = X
        self.output = np.dot(X, self.W) + self.b
        return self.output

    def backward(self, d_out, learning_rate, l2_lambda=0.0):
        """
        Backpropagation step:
        - d_out: gradient from next layer
        - Updates weights and biases using gradient descent
        """
        # Gradients
        dW = np.dot(self.input.T, d_out) + l2_lambda * self.W
        db = np.sum(d_out, axis=0, keepdims=True)

        # Gradient to propagate back to previous layer
        d_input = np.dot(d_out, self.W.T)

        # Update weights and biases
        self.W -= learning_rate * dW
        self.b -= learning_rate * db

        return d_input

