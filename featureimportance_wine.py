import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

df_wine=pd.read_csv('D:/Winter 2023/BSc DS 4/Programs/wine.csv')
print(df_wine)
feat_labels = df_wine.columns[1:]

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a decision tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Fit the classifier to the training data
clf.fit(X_train, y_train)

# Get feature importances
feature_importances = clf.feature_importances_

# Print feature importances
for feature, importance in zip(df_wine.columns, feature_importances):
    print(f"{feature}: {importance}")

indices = np.argsort(feature_importances)[::-1]
for f in range(X_train.shape[1]):print("%2d) %-*s %f" % (f + 1, 30,feat_labels[f],feature_importances[indices[f]]))

