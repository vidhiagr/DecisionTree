import numpy as np
import pandas as pd

train_data = pd.read_csv("Datasets/bank-note/train.csv", header=None)
test_data = pd.read_csv("Datasets/bank-note/test.csv", header=None)

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

y_train = np.where(y_train == 0, -1, 1)
y_test = np.where(y_test == 0, -1, 1)

# Hyperparametes
T = 100  # Max epochs
C_values = [100 / 873, 500 / 873, 700 / 873]  # Hyperparameter C
gamma_0_values = [0.01, 0.1, 1]  # learning rate candidates
a_values = [1, 10, 100]  # values for the learning rate schedule
gamma_schedule_1 = lambda gamma_0, a, t: gamma_0 / (1 + gamma_0 / a * t)
gamma_schedule_2 = lambda gamma_0, t: gamma_0 / (1 + t)

def hinge_loss(X, y, w, b, C):
    n = len(y)
    margins = 1 - y * (np.dot(X, w) + b)
    return 0.5 * np.dot(w, w) + C * np.sum(np.maximum(0, margins))

def svm_train(X, y, C, gamma_schedule, gamma_0, a=None):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0
    objectives = []

    for t in range(1, T + 1):
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        X, y = X[indices], y[indices]

        if a is not None:
            gamma_t = gamma_schedule(gamma_0, a, t)
        else:
            gamma_t = gamma_schedule(gamma_0, t)

        for i in range(n_samples):
            margin = y[i] * (np.dot(w, X[i]) + b)
            if margin < 1:
                w = w - gamma_t * w + gamma_t * C * y[i] * X[i]
                b = b + gamma_t * C * y[i]
            else:
                w = w - gamma_t * w

        objectives.append(hinge_loss(X, y, w, b, C))

    return w, b, objectives

def evaluate(X, y, w, b):
    predictions = np.sign(np.dot(X, w) + b)
    error = np.mean(predictions != y)
    return error

results = {}
for C in C_values:
    for gamma_0 in gamma_0_values:
        for a in a_values:
            # Schedule 1
            w1, b1, obj1 = svm_train(X_train, y_train, C, gamma_schedule_1, gamma_0, a)
            train_error_1 = evaluate(X_train, y_train, w1, b1)
            test_error_1 = evaluate(X_test, y_test, w1, b1)

            # Schedule 2
            w2, b2, obj2 = svm_train(X_train, y_train, C, gamma_schedule_2, gamma_0)
            train_error_2 = evaluate(X_train, y_train, w2, b2)
            test_error_2 = evaluate(X_test, y_test, w2, b2)

            results[(C, gamma_0, a)] = {
                "schedule_1": {
                    "w": w1,
                    "b": b1,
                    "train_error": train_error_1,
                    "test_error": test_error_1,
                    "objectives": obj1,
                },
                "schedule_2": {
                    "w": w2,
                    "b": b2,
                    "train_error": train_error_2,
                    "test_error": test_error_2,
                    "objectives": obj2,
                },
            }


# # Hyperparameters for Part (a)
# C_values = [100 / 873, 500 / 873, 700 / 873]  # Values of C
# gamma_0_values = [0.01, 0.1, 1]  # Initial learning rates
# a_values = [1, 10, 100]  # Values for 'a'

# # Run experiments for schedule 1
# results_schedule_1 = {}
# for C in C_values:
#     for gamma_0 in gamma_0_values:
#         for a in a_values:
#             w, b, objectives = svm_train(X_train, y_train, C, gamma_schedule_1, gamma_0, a)
#             train_error = evaluate(X_train, y_train, w, b)
#             test_error = evaluate(X_test, y_test, w, b)
#             results_schedule_1[(C, gamma_0, a)] = {
#                 "w": w,
#                 "b": b,
#                 "train_error": train_error,
#                 "test_error": test_error,
#                 "objectives": objectives,
#             }

# # Print Training and Test Errors for Schedule 1
# for key, result in results_schedule_1.items():
#     print(f"C={key[0]:.4f}, γ₀={key[1]}, a={key[2]} -> Train Error: {result['train_error']:.4f}, Test Error: {result['test_error']:.4f}")

# Hyperparameters for Part (b)
C_values = [100 / 873, 500 / 873, 700 / 873]  # Values of C
gamma_0_values = [0.01, 0.1, 1]  # Initial learning rates

# learning rate schedule for Part (b)
gamma_schedule_2 = lambda gamma_0, t: gamma_0 / (1 + t)

# experiments for schedule 2
results_schedule_2 = {}
for C in C_values:
    for gamma_0 in gamma_0_values:
        w, b, objectives = svm_train(X_train, y_train, C, gamma_schedule_2, gamma_0)
        train_error = evaluate(X_train, y_train, w, b)
        test_error = evaluate(X_test, y_test, w, b)
        results_schedule_2[(C, gamma_0)] = {
            "w": w,
            "b": b,
            "train_error": train_error,
            "test_error": test_error,
            "objectives": objectives,
        }

# Training and Test Errors for Schedule 2
for key, result in results_schedule_2.items():
    print(f"C={key[0]:.4f}, γ₀={key[1]} -> Train Error: {result['train_error']:.4f}, Test Error: {result['test_error']:.4f}")



