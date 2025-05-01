# 🧠 MNIST Neural Network

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/mnist_neural_network.git
cd mnist_neural_network
```

### 2. Create and Activate a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🧪 Usage
Run the main script to execute the pipeline:
```bash
python src/main.py
```

### What the Script Does
- **Loads and Preprocesses Data**: Downloads MNIST, normalizes data, splits into train/test, one-hot encodes
- **Visualizes Data**: Saves five random images to `random_images.png`
- **Runs Experiments** with:
  - Learning rates: `0.1`, `0.01`, `0.001`
  - Architectures:
    - `[784, 256, 128, 64, 10]`
    - `[784, 512, 256, 10]`
    - `[784, 10]`
    - `[784, 64, 10]`
    - `[784, 128, 10]`
  - Activation functions: `tanh`, `sigmoid`
  - Loss functions: `cross_entropy`, `mse`
- **Evaluates Performance**:
  - Finds best configuration
  - Prints confusion matrix, metrics, interpretation

---

## 📤 Output

### Console Output
- Training loss/accuracy per epoch
- Test accuracy for each experiment
- Final evaluation metrics

### File Output
- `random_images.png`: Random samples with labels
- `assets/Experiments_Screenshots/`:
  - `experiment_01.png`, ..., `experiment_05.png`
  - `confusion_matrix.png`
  - `model_evaluation_metrics.png`

---

## 📁 Project Structure
```bash
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
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── neural_network.py
│   ├── train.py
│   ├── evaluate.py
│   └── main.py
├── README.md
├── requirements.txt
└── LICENSE.txt
```

---

## 📊 Experiment Results

### Experiment 1: Architecture `[784-10]`, Learning Rate `0.001`
```
Epoch 1: Loss=3.3748 %, Accuracy=78.8580 %
Epoch 2: Loss=2.8467 %, Accuracy=83.2317 %
...
```

### Experiment 2: Architecture `[784-64-10]`, Learning Rate `0.001`
```
Epoch 1: Loss=4.4485 %, Accuracy=75.5558 %
...
```

### Experiment 3: Architecture `[784-512-256-10]`, Learning Rate `0.001`
```
Epoch 1: Loss=56.1384 %, Accuracy=69.1233 %
...
```

### Experiment 4: Architecture `[784-256-128-64-10]`, Learning Rate `0.001`
```
Epoch 1: Loss=3.3194 %, Accuracy=69.2658 %
...
```

### Experiment 5: Architecture `[784-256-128-64-10]`, Learning Rate `0.1`
```
Epoch 1: Loss=2.8952 %, Accuracy=80.8467 %
...
```

---

## 📈 Evaluation Results

**Best Configuration:**
- Learning Rate: `0.1`
- Architecture: `[784, 256, 128, 64, 10]`
- Activation: `tanh`
- Loss: `cross_entropy`

**Final Metrics:**
```yaml
Final Train Loss: 0.0456
Final Train Accuracy: 98.12%
Test Accuracy: 94.56%
```

**Confusion Matrix (sample):**
```text
[[ 950   0   2 ... ]
 [   0 1100   5 ... ]
 ... ]
```

- Accuracy: `94.56%`
- Precision (Macro Avg): `0.9452`
- Recall (Macro Avg): `0.9448`
- F1 Score (Macro Avg): `0.9450`

---

## 🤝 Contributing
Contributions are welcome!

### Steps to Contribute
1. Fork the repository
2. Create a new branch:
```bash
git checkout -b feature/your-feature
```
3. Commit your changes:
```bash
git commit -m "Add your feature"
```
4. Push to your branch:
```bash
git push origin feature/your-feature
```
5. Open a Pull Request

> Please follow PEP 8 standards and include documentation where needed.

---

## ❓ FAQ

### Why is the MNIST dataset downloaded automatically?
We use `fetch_openml` from `scikit-learn`, which downloads the dataset on first use.

### Can I modify the hyperparameters?
Yes. Edit `main.py` to change:
- `learning_rates`
- `architectures`
- `activation_functions`
- `loss_functions`
- `epochs`
- `batch_size`

### Why are there no test files?
This structure is simplified for clarity. You may add a `tests/` folder with `pytest` if desired.

### How can I save the trained model?
You can modify `train.py` to save weights using `np.save`. For help, open an issue or contact the maintainer.

---

## 🔗 References
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) by Michael Nielsen
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

---

## 📄 License
This project is licensed under the MIT License. See `LICENSE.txt` for details.

---

## 📬 Contact
For questions or suggestions:
- Open an issue on GitHub
- Contact: [Your Name] at [your.email@example.com]
