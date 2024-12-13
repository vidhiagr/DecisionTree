import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data(file_path):
    data = pd.read_csv(file_path, header=None)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    y = y.reshape(-1, 1)  
    return X, y

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

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

# Initialize Weights and Biases
# def initialize_weights(layer_sizes):
#     weights = [np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01 for i in range(len(layer_sizes)-1)]
#     biases = [np.zeros((1, layer_sizes[i+1])) for i in range(len(layer_sizes)-1)]
#     return weights, biases

# Initialize Weights and Biases with Zero
def initialize_weights(layer_sizes):
    weights = [np.zeros((layer_sizes[i], layer_sizes[i+1])) for i in range(len(layer_sizes)-1)]
    biases = [np.zeros((1, layer_sizes[i+1])) for i in range(len(layer_sizes)-1)]
    return weights, biases

def train_nn_sgd(X_train, y_train, X_test, y_test, hidden_widths, gamma_0, d, epochs):
    layer_sizes = [X_train.shape[1], hidden_widths, hidden_widths, 1]
    weights, biases = initialize_weights(layer_sizes)

    training_errors = []
    test_errors = []
    objective_function = []
    
    epsilon = 1e-10  

    for epoch in range(epochs):
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]

        for i in range(X_train.shape[0]):
            X_sample = X_train[i:i+1]
            y_sample = y_train[i:i+1]
            
            activations, z_values = forward_pass(X_sample, weights, biases)
            gradients_W, gradients_b = backward_pass(activations, z_values, y_sample, weights)

            gamma_t = gamma_0 / (1 + (gamma_0 / d) * epoch)

            for j in range(len(weights)):
                weights[j] -= gamma_t * gradients_W[j]
                biases[j] -= gamma_t * gradients_b[j]

        activations, _ = forward_pass(X_train, weights, biases)
        predictions = activations[-1]
        loss = np.mean(-(y_train * np.log(predictions + epsilon) + (1 - y_train) * np.log(1 - predictions + epsilon)))
        objective_function.append(loss)

        train_error = np.mean((predictions > 0.5) != y_train)
        activations_test, _ = forward_pass(X_test, weights, biases)
        test_error = np.mean((activations_test[-1] > 0.5) != y_test)

        training_errors.append(train_error)
        test_errors.append(test_error)

        print(f"Epoch {epoch}, Loss: {loss:.4f}, Training Error: {train_error:.4f}, Test Error: {test_error:.4f}")

    return weights, biases, training_errors, test_errors, objective_function



X_train, y_train = load_data("Datasets/bank-note/train.csv")
X_test, y_test = load_data("Datasets/bank-note/test.csv")
hidden_widths_list = [5, 10, 25, 50, 100]
gamma_0 = 0.1
d = 0.01
epochs = 50

for hidden_width in hidden_widths_list:
    print(f"\nTraining with Hidden Width: {hidden_width}")
    weights, biases, training_errors, test_errors, objective_function = train_nn_sgd(
        X_train, y_train, X_test, y_test, hidden_width, gamma_0, d, epochs
    )
    