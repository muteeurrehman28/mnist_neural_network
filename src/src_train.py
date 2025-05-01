import numpy as np
from src_neural_network import forward, backward, initialize_weights, cross_entropy_loss, mean_squared_error


def train(
    X_train,
    y_train,
    layer_sizes,
    epochs=10,
    batch_size=64,
    learning_rate=0.1,
    activation='tanh',
    loss_function='cross_entropy',
):
    """Train the neural network."""
    weights, biases = initialize_weights(layer_sizes)
    final_train_loss = None
    final_train_accuracy = None

    for epoch in range(epochs):
        permutation = np.random.permutation(X_train.shape[0])
        X_train_shuffled = X_train[permutation]
        y_train_shuffled = y_train[permutation]

        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train_shuffled[i : i + batch_size]
            y_batch = y_train_shuffled[i : i + batch_size]

            activations, pre_activations = forward(X_batch, weights, biases, activation)
            grads_W, grads_b = backward(
                X_batch, y_batch, activations, pre_activations, weights, biases, activation
            )

            for j in range(len(weights)):
                weights[j] -= learning_rate * grads_W[j]
                biases[j] -= learning_rate * grads_b[j]

        # Compute loss and accuracy
        activations_train, _ = forward(X_train, weights, biases, activation)
        A_final = activations_train[-1]

        if loss_function == 'mse':
            loss = mean_squared_error(y_train, A_final)
        else:
            loss = cross_entropy_loss(y_train, A_final)

        preds = np.argmax(A_final, axis=1)
        labels = np.argmax(y_train, axis=1)
        accuracy = np.mean(preds == labels)

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f} - Accuracy: {accuracy * 100:.2f}%")

        # Store final metrics
        if epoch == epochs - 1:
            final_train_loss = loss
            final_train_accuracy = accuracy * 100

    return weights, biases, final_train_loss, final_train_accuracy