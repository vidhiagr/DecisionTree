import numpy as np
import pandas as pd
from scipy.optimize import minimize

train_data = pd.read_csv("Datasets/bank-note/train.csv", header=None)
test_data = pd.read_csv("Datasets/bank-note/test.csv", header=None)

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

y_train = np.where(y_train == 0, -1, 1)
y_test = np.where(y_test == 0, -1, 1)

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def compute_gram_matrix(X):
    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i, j] = linear_kernel(X[i], X[j])
    return K


def objective(alpha, y, K, C):
    return 0.5 * np.sum(alpha[:, None] * alpha[None, :] * y[:, None] * y[None, :] * K) - np.sum(alpha)

def equality_constraint(alpha, y):
    return np.dot(alpha, y)

def get_bounds(C, n_samples):
    return [(0, C) for _ in range(n_samples)]

def recover_weights_and_bias(alpha, y, X):
    w = np.sum(alpha[:, None] * y[:, None] * X, axis=0)
    support_indices = np.where((alpha > 1e-5) & (alpha < C))[0]
    b = np.mean(y[support_indices] - np.dot(X[support_indices], w))
    return w, b

# Dual SVM Solver
def dual_svm(X, y, C):
    n_samples, n_features = X.shape
    K = compute_gram_matrix(X)
    
    initial_alpha = np.zeros(n_samples)

    constraints = [{'type': 'eq', 'fun': equality_constraint, 'args': (y,)}]
    bounds = get_bounds(C, n_samples)
 
    result = minimize(objective, initial_alpha, args=(y, K, C), 
                      method='SLSQP', bounds=bounds, constraints=constraints)
    
    alpha = result.x
    w, b = recover_weights_and_bias(alpha, y, X)
    return w, b, alpha


C_values = [100 / 873, 500 / 873, 700 / 873]
dual_results = {}

for C in C_values:
    w_dual, b_dual, alpha = dual_svm(X_train, y_train, C)
    dual_results[C] = {"w": w_dual, "b": b_dual}
    print(f"C={C:.4f}, w={w_dual}, b={b_dual}")

for C in C_values:
    print(f"\nComparison for C={C:.4f}")
    print(f"Dual SVM w: {dual_results[C]['w']}, b: {dual_results[C]['b']}")
    print(f"Primal SVM w and b ")
