import numpy as np
from scipy.optimize import minimize
import pandas as pd

train_data = pd.read_csv("Datasets/bank-note/train.csv", header=None)
test_data = pd.read_csv("Datasets/bank-note/test.csv", header=None)

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

y_train = np.where(y_train == 0, -1, 1)
y_test = np.where(y_test == 0, -1, 1)

def gaussian_kernel(x1, x2, gamma):
    return np.exp(-np.linalg.norm(x1 - x2) ** 2 / gamma)

def compute_gram_matrix(X, gamma):
    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i, j] = gaussian_kernel(X[i], X[j], gamma)
    return K

def objective(alpha, y, K, C):
    return 0.5 * np.sum(alpha[:, None] * alpha[None, :] * y[:, None] * y[None, :] * K) - np.sum(alpha)

def equality_constraint(alpha, y):
    return np.dot(alpha, y)

def get_bounds(C, n_samples):
    return [(0, C) for _ in range(n_samples)]

def predict(alpha, y, K, b, X_train, X_test):
    support_indices = np.where(alpha > 1e-5)[0]
    K_test = np.zeros((X_test.shape[0], len(support_indices)))
    for i in range(X_test.shape[0]):
        for j, sv_index in enumerate(support_indices):
            K_test[i, j] = gaussian_kernel(X_test[i], X_train[sv_index], gamma)
    return np.sign(np.sum(K_test * (alpha[support_indices] * y[support_indices])[:, None].T, axis=1) + b)

def dual_svm_gaussian(X, y, C, gamma):
    n_samples, n_features = X.shape
    K = compute_gram_matrix(X, gamma)
    
    initial_alpha = np.zeros(n_samples)
    
    constraints = [{'type': 'eq', 'fun': equality_constraint, 'args': (y,)}]
    bounds = get_bounds(C, n_samples)

    result = minimize(objective, initial_alpha, args=(y, K, C), 
                      method='SLSQP', bounds=bounds, constraints=constraints)
    
    alpha = result.x
    support_indices = np.where((alpha > 1e-5) & (alpha < C))[0]
    
    b = np.mean(
        y[support_indices] - np.dot(K[support_indices, :], (alpha * y))
    )
    return alpha, b

def evaluate(alpha, b, gamma, X_train, y_train, X_test, y_test):
  
    train_predictions = predict(alpha, y_train, compute_gram_matrix(X_train, gamma), b, X_train, X_train)
    train_error = np.mean(train_predictions != y_train)
    test_predictions = predict(alpha, y_train, compute_gram_matrix(X_train, gamma), b, X_train, X_test)
    test_error = np.mean(test_predictions != y_test)
    
    return train_error, test_error

C_values = [100 / 873, 500 / 873, 700 / 873]
gamma_values = [0.1, 0.5, 1, 5, 100]

results = {}
for C in C_values:
    for gamma in gamma_values:
        alpha, b = dual_svm_gaussian(X_train, y_train, C, gamma)
        train_error, test_error = evaluate(alpha, b, gamma, X_train, y_train, X_test, y_test)
        results[(C, gamma)] = {"train_error": train_error, "test_error": test_error}
        print(f"C={C:.4f}, gamma={gamma:.1f} -> Train Error: {train_error:.4f}, Test Error: {test_error:.4f}")

best_combination = min(results.items(), key=lambda x: x[1]['test_error'])
print("\nBest combination:")
print(f"C={best_combination[0][0]:.4f}, gamma={best_combination[0][1]:.1f} -> Train Error: {best_combination[1]['train_error']:.4f}, Test Error: {best_combination[1]['test_error']:.4f}")
