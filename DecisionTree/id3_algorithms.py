import pandas as pd
import numpy as np

# Information Gain
def entropy(y):
    values, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -np.sum(probs * np.log2(probs))

# def information_gain(X, y, feature):
#     parent_entropy = entropy(y)
#     values, counts = np.unique(X[feature], return_counts=True)
#     weighted_entropy = np.sum((counts / len(X)) * [entropy(y[X[feature] == v]) for v in values])
#     return parent_entropy - weighted_entropy

# Majority Error
def majority_error(y):
    values, counts = np.unique(y, return_counts=True)
    return 1 - np.max(counts) / len(y)

def majority_error_gain(X, y, feature):
    parent_error = majority_error(y)
    values, counts = np.unique(X[feature], return_counts=True)
    weighted_error = np.sum((counts / len(X)) * [majority_error(y[X[feature] == v]) for v in values])
    return parent_error - weighted_error

# Gini Index
def gini_index(y):
    values, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1 - np.sum(probs**2)

def gini_gain(X, y, feature):
    parent_gini = gini_index(y)
    values, counts = np.unique(X[feature], return_counts=True)
    weighted_gini = np.sum((counts / len(X)) * [gini_index(y[X[feature] == v]) for v in values])
    return parent_gini - weighted_gini

class DecisionTree:
    def __init__(self, max_depth, criterion):
        self.max_depth = max_depth
        self.criterion = criterion
    
    def fit(self, X, y, depth=0):
        # Base cases for recursion
        if len(np.unique(y)) == 1 or depth == self.max_depth:
            return np.unique(y)[0]
        
        if self.criterion == 'information_gain':
            gains = [information_gain(X, y, feature) for feature in X.columns]
        elif self.criterion == 'majority_error':
            gains = [majority_error_gain(X, y, feature) for feature in X.columns]
        elif self.criterion == 'gini_index':
            gains = [gini_gain(X, y, feature) for feature in X.columns]
        
        # Choose the best feature
        best_feature = X.columns[np.argmax(gains)]
        
        tree = {best_feature: {}}
        for value in np.unique(X[best_feature]):
            subtree = self.fit(X[X[best_feature] == value].drop([best_feature], axis=1),
                               y[X[best_feature] == value], depth + 1)
            tree[best_feature][value] = subtree
        return tree

    def predict(self, X, tree):
        for feature in tree.keys():
            value = X[feature]
            
            # case where the value is not present in the tree
            if value not in tree[feature]:
                if tree[feature]:
                    values = [v for v in tree[feature].values() if not isinstance(v, dict)]
                    if values:
                        return max(set(values), key=values.count) 
                    else:
                        return 0
                else:
                    return 0

            tree = tree[feature][value]
            if isinstance(tree, dict):
                return self.predict(X, tree)
            else:
                return tree
            
class DecisionStump:
    def __init__(self, criterion):
        self.criterion = criterion
        self.tree = None
    
    def fit(self, X, y, weights):
        # Base case, depth 1
        if self.criterion == 'information_gain':
            gains = [information_gain(X, y, feature, weights) for feature in X.columns]
        elif self.criterion == 'majority_error':
            gains = [majority_error_gain(X, y, feature, weights) for feature in X.columns]
        elif self.criterion == 'gini_index':
            gains = [gini_gain(X, y, feature, weights) for feature in X.columns]
        
        # Choose the best feature
        best_feature = X.columns[np.argmax(gains)]
        # print(f"Training stump with weights: {weights[:10]}")
        
        self.tree = {best_feature: {}}
        for value in np.unique(X[best_feature]):
            subtree = np.argmax(np.bincount(y[X[best_feature] == value], weights[X[best_feature] == value]))
            self.tree[best_feature][value] = subtree
    
    def predict(self, X):
        for feature in self.tree.keys():
            value = X[feature]  # Ensure X is a single row here
            return self.tree[feature].get(value, np.random.choice([0, 1]))  # Predict based on the tree


def weighted_entropy(y, weights):
    values, counts = np.unique(y, return_counts=True)
    weighted_counts = np.array([np.sum(weights[y == v]) for v in values])
    probs = weighted_counts / np.sum(weights)
    return -np.sum(probs * np.log2(probs))

def information_gain(X, y, feature, weights):
    parent_entropy = weighted_entropy(y, weights)
    values, counts = np.unique(X[feature], return_counts=True)
    weighted_entropy_val = np.sum((counts / len(X)) * [weighted_entropy(y[X[feature] == v], weights[X[feature] == v]) for v in values])
    return parent_entropy - weighted_entropy_val
