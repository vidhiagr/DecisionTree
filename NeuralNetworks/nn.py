import numpy as np
import pandas as pd


def load_data(file_path):
    data = pd.read_csv(file_path, header=None)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    y = y.reshape(-1, 1) 
    return X, y

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def forward_pass(X, weights, biases):
    activations = [X]
    z_values = []
    
    for W, b in zip(weights, biases):
        z = np.dot(activations[-1], W) + b
        z_values.append(z)
        activations.append(sigmoid(z))
    
    return activations, z_values


def backward_pass(activations, z_values, y, weights):
    deltas = [activations[-1] - y] 
    gradients_W = []
    gradients_b = []

    for i in range(len(weights) - 1, -1, -1):
        grad_W = np.dot(activations[i].T, deltas[-1])
        grad_b = np.sum(deltas[-1], axis=0, keepdims=True)
        gradients_W.insert(0, grad_W)
        gradients_b.insert(0, grad_b)
        
        if i > 0:
            delta = np.dot(deltas[-1], weights[i].T) * sigmoid_derivative(z_values[i - 1])
            deltas.append(delta)

    return gradients_W, gradients_b


def train_nn(X_train, y_train, hidden_layers, learning_rate, epochs):
    layer_sizes = [X_train.shape[1]] + hidden_layers + [1]
    weights = [np.random.randn(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)]
    biases = [np.zeros((1, layer_sizes[i+1])) for i in range(len(layer_sizes)-1)]
    
    for epoch in range(epochs):
        activations, z_values = forward_pass(X_train, weights, biases)
        gradients_W, gradients_b = backward_pass(activations, z_values, y_train, weights)
        
        for i in range(len(weights)):
            weights[i] -= learning_rate * gradients_W[i]
            biases[i] -= learning_rate * gradients_b[i]
        
        if epoch % 10 == 0:
            loss = np.mean(-(y_train * np.log(activations[-1]) + (1 - y_train) * np.log(1 - activations[-1])))
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    return weights, biases

X_train, y_train = load_data("Datasets/bank-note/train.csv")
X_test, y_test = load_data("Datasets/bank-note/test.csv")

weights, biases = train_nn(X_train, y_train, hidden_layers=[8, 4], learning_rate=0.01, epochs=100)
