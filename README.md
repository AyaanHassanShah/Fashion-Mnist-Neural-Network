# Fashion-MNIST Neural Network Assignment

This repository contains the implementation and experiments for a neural network assignment using the **Fashion-MNIST** dataset. The assignment covers neural networks from scratch, PyTorch-based models, activation functions, loss functions, optimizers, overfitting, regularization, and hyperparameter optimization.

## Dataset

The project uses the **Fashion-MNIST** dataset, which contains grayscale images of clothing items belonging to 10 different classes.

* Training samples: 60,000
* Test samples: 10,000
* Image size: 28 × 28 pixels
* Input features after flattening: 784
* Number of classes: 10

The training data was normalized to the range **0 to 1** and split into **80% training and 20% validation data**. The test set was kept separate and untouched until the final evaluation.

## Assignment Parts

### Part 1: Neural Network from Scratch

A neural network was implemented using **NumPy** without relying on an automatic differentiation framework.

Architecture:

```text
784 → 64 → 10
```

The implementation included:

* Forward propagation
* ReLU activation
* Softmax output
* Categorical Cross-Entropy loss
* Manual backpropagation
* Gradient descent
* Loss visualization
* Gradient verification using PyTorch

The manually calculated gradients were compared with PyTorch gradients to verify the correctness of the implementation.

### Part 2: Activation Functions

A deeper neural network was implemented using PyTorch. Four activation functions were compared:

* Sigmoid
* Tanh
* ReLU
* Leaky ReLU

Training and validation performance were analyzed using loss and accuracy curves. Gradient behavior was also investigated, including the vanishing gradient problem and dead ReLU units.

The ReLU model achieved approximately **88.46% validation accuracy** among the tested activation functions.

### Part 3: Loss Functions

The classification model was trained using two different loss functions:

* Categorical Cross-Entropy
* Mean Squared Error

Their training behavior and classification performance were compared.

A separate neural network was also trained for a tabular regression task. The regression model was evaluated using:

* MSE
* RMSE
* MAE

This part demonstrated why Cross-Entropy is generally more appropriate and efficient for multi-class classification.

### Part 4: Optimizer Comparison

Four optimization algorithms were compared:

* SGD
* SGD with Momentum
* RMSProp
* Adam

The experiments compared learning speed, validation accuracy, training time, and the number of epochs required to reach a target validation accuracy.

Different learning rates were also investigated to understand their effect on optimization performance.

### Part 5: Overfitting

An intentionally oversized neural network was trained on only **2,000 training samples** to demonstrate overfitting.

The model contained:

```text
784 → 512 → 512 → 512 → 512 → 10
```

The model was trained until the training accuracy exceeded 99%. The large difference between training and validation performance demonstrated **high variance and overfitting**.

Training and validation loss curves were used to identify when the model began to overfit.

### Part 6: Regularization

Several techniques were investigated to reduce the overfitting observed in Part 5:

* L2 regularization
* L1 regularization
* Dropout
* Batch Normalization
* Early Stopping
* Data Augmentation
* Increasing the amount of training data

Different regularization strengths and dropout rates were tested. Training accuracy, validation accuracy, and the generalization gap were used to compare the methods.

For L1 regularization, the percentage of model weights smaller than **1e-3** was also measured to evaluate sparsity.

### Part 7: Hyperparameter Search and Final Evaluation

The final part used **random hyperparameter search** with **5-fold cross-validation**.

Multiple hyperparameters were varied and the resulting configurations were ranked using their cross-validation performance.

The selected configuration was then retrained using the available training data and the chosen regularization technique.

The final model was evaluated on the previously untouched test set using:

* Accuracy
* Macro Precision
* Macro Recall
* Macro F1-score
* Confusion Matrix

The final results were compared with the Part 2 baseline to measure the improvement obtained through optimization and regularization.

## Technologies Used

* Python
* NumPy
* PyTorch
* Pandas
* Matplotlib
* Scikit-learn
* Jupyter Notebook
* Kaggle GPU environment

## Model Development Workflow

The assignment followed this general workflow:

```text
Fashion-MNIST
      ↓
Data Preprocessing
      ↓
Train / Validation Split
      ↓
Neural Network from Scratch
      ↓
PyTorch Neural Network
      ↓
Activation Function Comparison
      ↓
Loss Function Comparison
      ↓
Optimizer Comparison
      ↓
Intentional Overfitting
      ↓
Regularization
      ↓
Hyperparameter Search
      ↓
Final Test Evaluation
```

## Repository Structure

```text
fashion-mnist-neural-network/
│
├── README.md
├── notebooks/
│   └── fashion_mnist_assignment.ipynb
│
├── results/
│   ├── plots/
│   └── tables/
│
└── requirements.txt
```

## Installation

Clone the repository:

```bash
git clone https://github.com/USERNAME/fashion-mnist-neural-network.git
cd fashion-mnist-neural-network
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Running the Project

The experiments were developed and executed in a **Kaggle notebook using GPU acceleration**.

Open the notebook:

```text
notebooks/fashion_mnist_assignment.ipynb
```

and run the cells sequentially from Part 1 through Part 7.

## Results

The project demonstrates the complete neural network development process, starting from a manually implemented NumPy network and progressing to PyTorch models, optimization, overfitting analysis, regularization, and final hyperparameter tuning.

Key observations include:

* Manual backpropagation can be verified against automatic differentiation.
* Activation functions significantly affect training and generalization.
* Cross-Entropy is better suited to multi-class classification than MSE.
* Advanced optimizers can converge faster than basic SGD.
* Large models can easily overfit small datasets.
* Regularization can reduce the generalization gap.
* Hyperparameter tuning and cross-validation can improve model selection.

## Author

**Syed Ayaan Hassan Shah**
**Muhammad Luqman Waseem**

FAST National University of Computer and Emerging Sciences

## License

This project was created for academic purposes.
