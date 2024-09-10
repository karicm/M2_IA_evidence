# Author: Karla Stefania Cruz Muñiz
# Date: 09.09.2024
# Code for random forest with a dataset of mobile phone prices.

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
# import matplotlib.pyplot as plt

# Load the training data
train_df = pd.read_csv('train.csv')

# Separate input data (X) from the target data (y)
X = train_df.drop('price_range', axis=1)
y = train_df['price_range']

# Split the training set into training and validation (80% - 20%)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the Random Forest model
model = RandomForestClassifier(random_state=40, n_estimators=300, max_depth=17)

# Train the model with the training data
model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = model.predict(X_val)

# Calculate the accuracy
accuracy = accuracy_score(y_val, y_pred)
print("\nAccuracy in validation data:", accuracy)

# Print confusion matrix
conf_matrix = confusion_matrix(y_val, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Plot confusion matrix
# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='RdPu', 
#             xticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3'], 
#             yticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3'])
# plt.xlabel('Prediction')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()

# Classification report
class_report = classification_report(y_val, y_pred)
print("\nClassification Report:")
print(class_report)

# Calculate ROC-AUC for each class
roc_auc = roc_auc_score(y_val, model.predict_proba(X_val), multi_class='ovr')
print("ROC-AUC:", roc_auc)

# Perform cross-validation with 7 folds
cv = StratifiedKFold(n_splits=7, shuffle=True, random_state=42)
accuracy_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

# Calculate the average accuracy in the folds
average_accuracy = accuracy_scores.mean()
print("\nAverage accuracy with cross-validation:", average_accuracy)

# Feature Importance Analysis
importances = model.feature_importances_
indices = importances.argsort()[::-1]
features = X.columns

print("\nFeature importance:")
for i in range(X.shape[1]):
    print(f"- {features[indices[i]]}: {importances[indices[i]]}")
