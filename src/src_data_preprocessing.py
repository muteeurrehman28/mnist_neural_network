import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def load_mnist_data():
    """Load and preprocess MNIST dataset."""
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist['data']
    y = mnist['target'].astype(int)
    return X, y


def normalize_data(X):
    """Normalize pixel values to [0, 1]."""
    return X / 255.0


def split_data(X, y, train_size=60000, test_size=10000, random_state=42):
    """Split data into training and test sets."""
    return train_test_split(
        X, y, train_size=train_size, test_size=test_size, random_state=random_state
    )


def one_hot_encode(y, num_classes=10):
    """Convert labels to one-hot encoded format."""
    one_hot = np.zeros((y.size, num_classes))
    one_hot[np.arange(y.size), y] = 1
    return one_hot


def display_random_images(X, y, n=5):
    """Display n random images from the dataset."""
    plt.figure(figsize=(10, 2))
    for i in range(n):
        index = np.random.randint(0, X.shape[0])
        image = X[index].reshape(28, 28)
        label = y[index]
        plt.subplot(1, n, i + 1)
        plt.imshow(image, cmap='gray')
        plt.title(f"Label: {label}")
        plt.axis('off')
    plt.savefig('random_images.png')
    plt.close()