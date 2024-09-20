import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from id3_algorithms import DecisionTree


# Dataset path
train_file = 'bank/train.csv'
test_file = 'bank/test.csv'

# Column Names
column_names = ['age', 'job', 'marital', 'education', 'default', 'balance', 
                'housing', 'loan', 'contact', 'day', 'month', 'duration', 
                'campaign', 'pdays', 'previous', 'poutcome', 'y']

# Load data
train_data = pd.read_csv(train_file, header=None, names=column_names)
test_data = pd.read_csv(test_file, header=None, names=column_names)

# replace 'unknown' with the majority value 
def replace_unknown_with_majority(data):
    for col in data.columns:
        if 'unknown' in data[col].values:
            majority_value = data.loc[data[col] != 'unknown', col].mode()[0]
            data[col].replace('unknown', majority_value, inplace=True)
    return data

train_data = replace_unknown_with_majority(train_data)

# Encode data
categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 
                       'loan', 'contact', 'month', 'poutcome', 'y']

for col in categorical_columns:
    train_data[col] = train_data[col].astype('category').cat.codes
    test_data[col] = test_data[col].astype('category').cat.codes


# Split data
X_train = train_data.drop(columns=['y'])
y_train = train_data['y']
X_test = test_data.drop(columns=['y'])
y_test = test_data['y']

depth_range = range(1, 17)
criteria = ['information_gain', 'majority_error', 'gini_index']

results = []

for criterion in criteria:
    print(f"Evaluating criterion: {criterion}")
    for depth in depth_range:
        # Train
        tree = DecisionTree(max_depth=depth, criterion=criterion)
        tree_model = tree.fit(X_train, y_train)
        
        # Predict
        train_predictions = [tree.predict(row, tree_model) for _, row in X_train.iterrows()]
        test_predictions = [tree.predict(row, tree_model) for _, row in X_test.iterrows()]
        
        # Evaluate
        train_error = 1 - accuracy_score(y_train, train_predictions)
        test_error = 1 - accuracy_score(y_test, test_predictions)
        
        results.append((criterion, depth, train_error, test_error))
        print(f"Depth: {depth}, Train Error: {train_error}, Test Error: {test_error}")


