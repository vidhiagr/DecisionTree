import numpy as np
import pandas as pd

def load_data(file_path):
    data = pd.read_csv(file_path, header=None)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    y = 2 * y - 1  
    return X, y

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_loss(X, y, w, v):
    linear_term = np.dot(X, w)
    log_likelihood = np.sum(np.log(1 + np.exp(-y * linear_term)))
    regularization = (1 / (2 * v)) * np.sum(w ** 2)
    return log_likelihood + regularization

def compute_gradient(X, y, w, v):
    linear_term = np.dot(X, w)
    sigmoid_term = sigmoid(-y * linear_term)
    grad_log_likelihood = -np.dot(X.T, y * sigmoid_term)
    grad_regularization = w / v
    return grad_log_likelihood + grad_regularization

def logistic_regression_sgd(X_train, y_train, X_test, y_test, v_list, gamma_0, d, epochs):
    n_samples, n_features = X_train.shape
    results = []

    for v in v_list:
        w = np.zeros(n_features)
        training_errors = []
        test_errors = []
        losses = []

        for epoch in range(epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_train = X_train[indices]
            y_train = y_train[indices]

            for t in range(1, n_samples + 1):
                gamma_t = gamma_0 / (1 + (gamma_0 / d) * t)

                idx = t % n_samples
                x_i = X_train[idx].reshape(1, -1)
                y_i = y_train[idx]
                grad = compute_gradient(x_i, y_i, w, v)

                w -= gamma_t * grad

            loss = compute_loss(X_train, y_train, w, v)
            losses.append(loss)

            train_predictions = np.sign(np.dot(X_train, w))
            train_error = np.mean(train_predictions != y_train)
            training_errors.append(train_error)

            test_predictions = np.sign(np.dot(X_test, w))
            test_error = np.mean(test_predictions != y_test)
            test_errors.append(test_error)

        results.append((v, training_errors[-1], test_errors[-1]))


    return results


X_train, y_train = load_data("Datasets/bank-note/train.csv")
X_test, y_test = load_data("Datasets/bank-note/test.csv")

v_list = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]
gamma_0 = 0.1
d = 0.01
epochs = 100

results = logistic_regression_sgd(X_train, y_train, X_test, y_test, v_list, gamma_0, d, epochs)

print("Variance (v), Training Error, Test Error")
for v, train_err, test_err in results:
    print(f"{v:.2f}, {train_err:.4f}, {test_err:.4f}")
