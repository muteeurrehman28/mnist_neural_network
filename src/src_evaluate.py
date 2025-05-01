import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from src_neural_network import forward


def predict(X, weights, biases, activation='tanh'):
    """Predict labels for input data."""
    activations, _ = forward(X, weights, biases, activation)
    return np.argmax(activations[-1], axis=1)


def evaluate_model(y_test, predictions):
    """Compute and print evaluation metrics."""
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, predictions)
    print("\nConfusion Matrix:")
    print(conf_matrix)

    # Accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"\nAccuracy: {accuracy * 100:.2f}%")

    # Precision, Recall, F1-Score
    precision = precision_score(y_test, predictions, average='macro')
    recall = recall_score(y_test, predictions, average='macro')
    f1 = f1_score(y_test, predictions, average='macro')

    print(f"\nPrecision (Macro Average): {precision:.4f}")
    print(f"Recall (Macro Average): {recall:.4f}")
    print(f"F1 Score (Macro Average): {f1:.4f}")

    # Interpretation
    print("\nInterpretation:")
    print("Macro averaging calculates metrics for each class separately, then averages them equally.")
    print("High Precision: Fewer false positives.")
    print("High Recall: Fewer false negatives.")
    print("High F1-Score: Good balance of Precision and Recall.")

    return conf_matrix, accuracy, precision, recall, f1