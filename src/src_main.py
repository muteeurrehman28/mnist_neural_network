import numpy as np
from src_data_preprocessing import load_mnist_data, normalize_data, split_data, one_hot_encode, display_random_images
from src_train import train
from src_evaluate import predict, evaluate_model


def run_experiments():
    """Run experiments with different configurations."""
    # Load and preprocess data
    X, y = load_mnist_data()
    X = normalize_data(X)
    X_train, X_test, y_train, y_test = split_data(X, y)
    y_train_encoded = one_hot_encode(y_train)
    y_test_encoded = one_hot_encode(y_test)

    # Display random images
    display_random_images(X, y)

    # Experiment configurations
    learning_rates = [0.1, 0.01, 0.001]
    architectures = [
        [784, 256, 128, 64, 10],
        [784, 512, 256, 10],
        [784, 10],
        [784, 64, 10],
        [784, 128, 10],
    ]
    activation_functions = ['tanh', 'sigmoid']
    loss_functions = ['cross_entropy', 'mse']
    epochs = 10
    batch_size = 64

    results = []

    print("Running Experiments...")
    for lr in learning_rates:
        for arch in architectures:
            for act in activation_functions:
                for loss_fn in loss_functions:
                    print(f"\nExperiment: LR={lr}, Arch={arch}, Act={act}, Loss={loss_fn}")
                    weights, biases, train_loss, train_acc = train(
                        X_train,
                        y_train_encoded,
                        layer_sizes=arch,
                        epochs=epochs,
                        batch_size=batch_size,
                        learning_rate=lr,
                        activation=act,
                        loss_function=loss_fn,
                    )
                    predictions = predict(X_test, weights, biases, activation=act)
                    test_acc = np.mean(predictions == y_test) * 100
                    results.append(
                        {
                            'learning_rate': lr,
                            'architecture': arch,
                            'activation': act,
                            'loss_function': loss_fn,
                            'test_accuracy': test_acc,
                            'final_train_loss': train_loss,
                            'final_train_accuracy': train_acc,
                        }
                    )
                    print(f"Test Accuracy: {test_acc:.2f}%")

    # Evaluate best model
    best_result = max(results, key=lambda x: x['test_accuracy'])
    print(f"\nBest Configuration: LR={best_result['learning_rate']}, "
          f"Arch={best_result['architecture']}, Act={best_result['activation']}, "
          f"Loss={best_result['loss_function']}")
    print(f"Final Train Loss: {best_result['final_train_loss']:.4f}")
    print(f"Final Train Accuracy: {best_result['final_train_accuracy']:.2f}%")
    print(f"Test Accuracy: {best_result['test_accuracy']:.2f}%")

    # Retrain and evaluate best model
    weights, biases, _, _ = train(
        X_train,
        y_train_encoded,
        layer_sizes=best_result['architecture'],
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=best_result['learning_rate'],
        activation=best_result['activation'],
        loss_function=best_result['loss_function'],
    )
    predictions = predict(X_test, weights, biases, activation=best_result['activation'])
    evaluate_model(y_test, predictions)


if __name__ == "__main__":
    run_experiments()
