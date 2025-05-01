import numpy as np


def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    """Derivative of sigmoid function."""
    s = sigmoid(x)
    return s * (1 - s)


def tanh_derivative(x):
    """Derivative of tanh function."""
    return 1 - np.tanh(x) ** 2


def softmax(x):
    """Softmax activation function."""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def mean_squared_error(y_true, y_pred):
    """Mean squared error loss."""
    return np.mean((y_true - y_pred) ** 2)


def cross_entropy_loss(y_true, y_pred):
    """Cross-entropy loss."""
    epsilon = 1e-10
    y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


def initialize_weights(layer_sizes):
    """Initialize weights and biases using Xavier initialization."""
    np.random.seed(42)
    weights = []
    biases = []
    for i in range(len(layer_sizes) - 1):
        W = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(
            2.0 / (layer_sizes[i] + layer_sizes[i + 1])
        )
        b = np.zeros((1, layer_sizes[i + 1]))
        weights.append(W)
        biases.append(b)
    return weights, biases


def forward(X, weights, biases, activation='tanh'):
    """Forward propagation through the network."""
    activations = [X]
    pre_activations = []

    for i in range(len(weights)):
        Z = np.dot(activations[-1], weights[i]) + biases[i]
        pre_activations.append(Z)
        if i < len(weights) - 1:  # Hidden layers
            A = np.tanh(Z) if activation == 'tanh' else sigmoid(Z)
        else:  # Output layer
            A = softmax(Z)
        activations.append(A)

    return activations, pre_activations


def backward(X, y_true, activations, pre_activations, weights, biases, activation='tanh'):
    """Backward propagation to compute gradients."""
    m = y_true.shape[0]
    grads_W = []
    grads_b = []

    # Output layer (softmax + cross-entropy)
    dZ = activations[-1] - y_true
    dW = np.dot(activations[-2].T, dZ) / m
    db = np.sum(dZ, axis=0, keepdims=True) / m
    grads_W.append(dW)
    grads_b.append(db)

    # Hidden layers
    for i in range(len(weights) - 1, 0, -1):
        dA = np.dot(dZ, weights[i].T)
        dZ = dA * (
            tanh_derivative(pre_activations[i - 1])
            if activation == 'tanh'
            else sigmoid_derivative(pre_activations[i - 1])
        )
        dW = np.dot(activations[i - 1].T, dZ) / m
        db = np.sum(dZ, axis=0, keepdims=True) / m
        grads_W.append(dW)
        grads_b.append(db)

    # Reverse gradients to match layer order
    grads_W = grads_W[::-1]
    grads_b = grads_b[::-1]

    return grads_W, grads_b