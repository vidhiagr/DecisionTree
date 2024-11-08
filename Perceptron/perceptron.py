import numpy as np
import pandas as pd

def load_data(file_path):
    data = pd.read_csv(file_path, header=None)
    X = data.iloc[:, :-1].values 
    y = data.iloc[:, -1].values   
    y = np.where(y == 0, -1, 1)   
    return X, y

def standard_perceptron(X, y, T=10):
    w = np.zeros(X.shape[1])  
    for epoch in range(T):
        for i in range(len(y)):
            if y[i] * np.dot(X[i], w) <= 0:
                w = w + y[i] * X[i]  #
    return w

def predict(X, w):
    return np.sign(np.dot(X, w))

def calculate_error(X, y, w):
    predictions = predict(X, w)
    error = np.mean(predictions != y)
    return error

def voted_perceptron(X, y, T=10):
    w = np.zeros(X.shape[1])
    weights_counts = []
    count = 1
    
    for epoch in range(T):
        for i in range(len(y)):
            if y[i] * np.dot(X[i], w) <= 0:
                weights_counts.append((w.copy(), count))
                w = w + y[i] * X[i]
                count = 1
            else:
                count += 1
    weights_counts.append((w.copy(), count))
    return weights_counts

def predict_voted(X, weights_counts):
    predictions = []
    for x in X:
        vote = sum(count * np.sign(np.dot(w, x)) for w, count in weights_counts)
        predictions.append(np.sign(vote))
    return np.array(predictions)

def averaged_perceptron(X, y, T=10):
    w = np.zeros(X.shape[1])
    w_avg = np.zeros(X.shape[1])
    
    for epoch in range(T):
        for i in range(len(y)):
            if y[i] * np.dot(X[i], w) <= 0:
                w = w + y[i] * X[i]
            w_avg += w  
    return w_avg / (len(y) * T)  

X_train, y_train = load_data("Datasets/bank-note/train.csv")
X_test, y_test = load_data("Datasets/bank-note/test.csv")

# Run standard Perceptron
T = 10
w_std = standard_perceptron(X_train, y_train, T)
test_error_std = calculate_error(X_test, y_test, w_std)

print("Standard Perceptron Weights:", w_std)
print("Standard Perceptron Test Error:", test_error_std)

# Run voted Perceptron
weights_counts = voted_perceptron(X_train, y_train, T)
voted_predictions = predict_voted(X_test, weights_counts)
test_error_voted = np.mean(voted_predictions != y_test)

print("Voted Perceptron Weights and Counts:")
for w, c in weights_counts:
    print("Weight Vector:", w, "Count:", c)
print("Voted Perceptron Test Error:", test_error_voted)


# Run averaged Perceptron
w_avg = averaged_perceptron(X_train, y_train, T)
test_error_avg = calculate_error(X_test, y_test, w_avg)

print("Averaged Perceptron Weight Vector:", w_avg)
print("Averaged Perceptron Test Error:", test_error_avg)
