MNIST Neural Network
This project implements a fully connected neural network from scratch to classify handwritten digits from the MNIST dataset. The implementation is written in Python and supports customizable network architectures, activation functions, loss functions, and hyperparameters. The project is designed to be modular, well-documented, and suitable for educational purposes or as a starting point for further experimentation.
Project Description
The MNIST Neural Network project aims to classify handwritten digits (0-9) using a custom-built neural network. The MNIST dataset, a benchmark in machine learning, contains 70,000 grayscale images (28x28 pixels) with corresponding labels. This project implements the neural network without relying on high-level frameworks like TensorFlow or PyTorch, providing insight into the mechanics of neural networks, including forward and backward propagation, weight initialization, and gradient descent.
Key Features

Data Preprocessing: Loads, normalizes, and splits the MNIST dataset into training (60,000 samples) and test (10,000 samples) sets. Includes one-hot encoding for labels and visualization of random images.
Neural Network Implementation: Supports multiple hidden layers, tanh and sigmoid activation functions for hidden layers, and softmax for the output layer. Implements Xavier weight initialization and mini-batch gradient descent.
Loss Functions: Supports cross_entropy (with softmax) and mean_squared_error loss functions.
Training: Trains the network with configurable learning rates, batch sizes, epochs, and architectures. Prints training loss and accuracy per epoch.
Experiments: Runs experiments with different learning rates (0.1, 0.01, 0.001), architectures (e.g., [784, 256, 128, 64, 10], [784, 10]), activation functions, and loss functions to identify the best configuration based on test accuracy.
Evaluation: Evaluates the best model using a confusion matrix, accuracy, precision, recall, and F1-score (macro-averaged). Provides interpretation of metrics.
Modularity: Code is organized into modules for data preprocessing, neural network operations, training, and evaluation, making it easy to extend or modify.

Objectives

Demonstrate a complete neural network implementation from scratch.
Explore the impact of hyperparameters and architectural choices on model performance.
Provide a clear, reproducible codebase for learning and experimentation.

Requirements

Python 3.8 or higher
Dependencies listed in requirements.txt:
numpy>=1.21.0
matplotlib>=3.4.0
scikit-learn>=1.0.0



Installation

Clone the Repository:git clone https://github.com/yourusername/mnist_neural_network.git
cd mnist_neural_network


Create and Activate a Virtual Environment:python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:pip install -r requirements.txt



Usage
Run the main script to execute the entire pipeline, including data preprocessing, experiments, and evaluation:
python src/main.py

What the Script Does

Loads and Preprocesses Data: Downloads the MNIST dataset, normalizes pixel values to [0, 1], splits data into training and test sets, and one-hot encodes labels.
Visualizes Data: Saves a plot of five random MNIST images with labels (random_images.png).
Runs Experiments: Trains the neural network with different configurations:
Learning rates: [0.1, 0.01, 0.001]
Architectures: [[784, 256, 128, 64, 10], [784, 512, 256, 10], [784, 10], [784, 64, 10], [784, 128, 10]]
Activation functions: [tanh, sigmoid]
Loss functions: [cross_entropy, mse]


Evaluates Performance: Identifies the configuration with the highest test accuracy and retrains the model with that configuration.
Detailed Evaluation: Computes and displays the confusion matrix, accuracy, precision, recall, and F1-score for the test set, along with an interpretation of the metrics.

Output

Console Output: Training progress (loss and accuracy per epoch), test accuracy for each experiment, and detailed evaluation metrics for the best model.
File Output: random_images.png containing five random MNIST images.

Project Structure
mnist_neural_network/
├── assets/
│   └── Experiments_Screenshots/
│       ├── experiment_01.png
│       ├── experiment_02.png
│       ├── experiment_03.png
│       ├── experiment_04.png
│       ├── experiment_05.png
│       ├── confusion_matrix.png
│       └── model_evaluation_metrics.png
├── src/
│   ├── __init__.py              # Makes src a Python package
│   ├── data_preprocessing.py    # Data loading, normalization, splitting, and visualization
│   ├── neural_network.py        # Neural network functions (activations, forward/backward prop)
│   ├── train.py                # Training logic with mini-batch gradient descent
│   ├── evaluate.py             # Evaluation metrics (confusion matrix, precision, recall, F1)
│   ├── main.py                 # Main script to run experiments and evaluation
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
└── LICENSE.txt                  # MIT license

Experiment Results
Below are the results of training the neural network with different configurations, showing the training progress (loss and accuracy over epochs) and predictions on test images. Each experiment includes five sample predictions with their predicted and true labels, marked as correct (✓) or incorrect (✗).
Experiment 1: Architecture [784-10], Learning Rate 0.001

Epoch 1: Loss=3.3748 %, Accuracy=78.8580 %
Epoch 2: Loss=2.8467 %, Accuracy=83.2317 %
Epoch 3: Loss=2.3183 %, Accuracy=85.2283 %
Epoch 4: Loss=2.1476 %, Accuracy=86.2767 %
Epoch 5: Loss=1.8066 %, Accuracy=87.9843 %


Experiment 2: Architecture [784-64-10], Learning Rate 0.001

Epoch 1: Loss=4.4485 %, Accuracy=75.5558 %
Epoch 2: Loss=2.6653 %, Accuracy=81.5288 %
Epoch 3: Loss=2.8212 %, Accuracy=84.3817 %
Epoch 4: Loss=2.6490 %, Accuracy=86.6380 %
Epoch 5: Loss=1.9181 %, Accuracy=86.9433 %


Experiment 3: Architecture [784-512-256-10], Learning Rate 0.001

Epoch 1: Loss=56.1384 %, Accuracy=69.1233 %
Epoch 2: Loss=56.4342 %, Accuracy=78.8809 %
Epoch 3: Loss=46.7798 %, Accuracy=85.3517 %
Epoch 4: Loss=41.8479 %, Accuracy=87.0643 %


Experiment 4: Architecture [784-256-128-64-10], Learning Rate 0.001

Epoch 1: Loss=3.3194 %, Accuracy=69.2658 %
Epoch 2: Loss=2.2203 %, Accuracy=77.1917 %
Epoch 3: Loss=2.7976 %, Accuracy=81.1817 %
Epoch 4: Loss=2.3941 %, Accuracy=83.4050 %
Epoch 5: Loss=1.9637 %, Accuracy=85.1750 %


Experiment 5: Architecture [784-256-128-64-10], Learning Rate 0.1

Epoch 1: Loss=2.8952 %, Accuracy=80.8467 %
Epoch 2: Loss=1.7265 %, Accuracy=88.5558 %
Epoch 3: Loss=1.7347 %, Accuracy=88.2558 %
Epoch 4: Loss=1.5197 %, Accuracy=89.8117 %
Epoch 5: Loss=1.3958 %, Accuracy=90.5080 %


Evaluation Results
The best model (based on test accuracy) was evaluated with the following metrics:

Confusion Matrix: Visualizes the model's performance across all classes.
Model Evaluation Metrics: Includes accuracy, precision, recall, and F1-score (macro-averaged).



Example Results
Running python src/main.py may produce output like:
Running Experiments...
Experiment: LR=0.1, Arch=[784, 256, 128, 64, 10], Act=tanh, Loss=cross_entropy
Epoch 1/10 - Loss: 0.1234 - Accuracy: 85.67%
...
Test Accuracy: 92.34%
...
Best Configuration: LR=0.1, Arch=[784, 256, 128, 64, 10], Act=tanh, Loss=cross_entropy
Final Train Loss: 0.0456
Final Train Accuracy: 98.12%
Test Accuracy: 94.56%

Confusion Matrix:
[[ 950   0   2 ... ]
 [   0 1100   5 ... ]
 ... ]

Accuracy: 94.56%
Precision (Macro Average): 0.9452
Recall (Macro Average): 0.9448
F1 Score (Macro Average): 0.9450

Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a new branch (git checkout -b feature/your-feature).
Make your changes and commit (git commit -m "Add your feature").
Push to the branch (git push origin feature/your-feature).
Open a pull request.

Please ensure your code follows PEP 8 standards and includes appropriate documentation.
FAQ

Why is the MNIST dataset downloaded automatically?The fetch_openml function from scikit-learn downloads the dataset on first use, ensuring no manual data setup is needed.
Can I modify the hyperparameters?Yes, edit the main.py file to adjust learning_rates, architectures, activation_functions, loss_functions, epochs, or batch_size.
Why are there no test files?This is a simplified structure for clarity. You can add a tests/ directory with unit tests using pytest if needed.
How can I save the trained model?Add code to train.py to save weights and biases using np.save. Contact the maintainer for guidance.

References

MNIST Dataset: http://yann.lecun.com/exdb/mnist/
Neural Networks and Deep Learning by Michael Nielsen: http://neuralnetworksanddeeplearning.com/
Scikit-learn Documentation: https://scikit-learn.org/stable/

License
This project is licensed under the MIT License. See the LICENSE.txt file for details.
Contact
For questions or suggestions, open an issue on GitHub or contact [Your Name] at [your.email@example.com].
