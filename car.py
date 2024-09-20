import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from id3_algorithms import DecisionTree, information_gain, majority_error, gini_index

# Dataset path
train_file = "car/train.csv"
test_file = "car/test.csv"

# Read data 
train_data = pd.read_csv(train_file, header=None)
test_data = pd.read_csv(test_file,header=None)

# Assigning the column names
column_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label']
train_data.columns = column_names
test_data.columns = column_names

# Encoding data
combined_data = pd.concat([train_data, test_data], axis=0)

for col in combined_data.columns:
    combined_data[col] = combined_data[col].astype('category').cat.codes

train_data = combined_data.iloc[:len(train_data)]
test_data = combined_data.iloc[len(train_data):]

# Split dataset
X_train = train_data.drop(columns=['label'])
y_train = train_data['label']

X_test = test_data.drop(columns=['label'])
y_test = test_data['label']


depth_range = range(1, 7)
criteria = ['information_gain', 'majority_error', 'gini_index']

for criterion in criteria:
    print(f"\nEvaluating criterion: {criterion}")
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
        
        print(f"Depth: {depth}, Train Error: {train_error}, Test Error: {test_error}")
