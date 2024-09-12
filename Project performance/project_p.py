import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the training data
train_df = pd.read_csv('train.csv')

# Separate input data (X) from the target data (y)
X = train_df.drop('price_range', axis=1)
y = train_df['price_range']

# Split the training set into training and validation (80% - 20%)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the first rows of the training and validation sets
print("\nFirst rows of the training set (X_train):")
print(X_train.head())
print("\nFirst rows of the validation set (X_val):")
print(X_val.head())

# Proportion of classes in the training and validation sets
print("\nDistribution of classes in the training set:")
print(y_train.value_counts(normalize=True))
print("\nDistribution of classes in the validation set:")
print(y_val.value_counts(normalize=True))

# Create a plot to visualize the distribution of classes in the training and validation sets
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.countplot(x=y_train, ax=axes[0], palette='viridis')
axes[0].set_title('Class Distribution - Training set')
axes[0].set_xlabel('Class')
axes[0].set_ylabel('Amount')

sns.countplot(x=y_val, ax=axes[1], palette='plasma')
axes[1].set_title('Class Distribution - Validation set')
axes[1].set_xlabel('Class')
axes[1].set_ylabel('Amount')

plt.tight_layout()
plt.show()

# Create the Random Forest model
model = RandomForestClassifier(random_state=40, n_estimators=300, max_depth=7, min_samples_leaf=4)

# Train the model with the training data
model.fit(X_train, y_train)

# Predict with the validation set
y_pred = model.predict(X_val)

# Predict with the training set (for comparison)
y_train_pred = model.predict(X_train)

# Calculate the accuracy in the validation and training sets
accuracy_val = accuracy_score(y_val, y_pred)
accuracy_train = accuracy_score(y_train, y_train_pred)
print("Accuracy in validation data:", accuracy_val)
print("Accuracy in training data:", accuracy_train)

# Print confusion matrix
conf_matrix = confusion_matrix(y_val, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='RdPu',
            xticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3'],
            yticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3'])
plt.xlabel('Prediction')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Generate learning curves to diagnose bias and variance
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(model, X, y, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10))

# Calculate the mean and standard deviation of the scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

# Plot the learning curves
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, 'o-', color='hotpink', label='Training Accuracy')
plt.plot(train_sizes, val_mean, 'o-', color='magenta', label='Validation Accuracy')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='hotpink', alpha=0.1)
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, color='magenta', alpha=0.1)
plt.title('Learning Curves')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.grid()
plt.show()

# Diagnose the degree of bias and variance
print("\nDiagnosing the degree of bias and variance:")

# If the training and validation accuracy are low, there is likely high bias
if accuracy_train < 0.8 and accuracy_val < 0.8:
    print("The model has high bias (underfitting). Recommendation: increase the complexity of the model or add more features.")

# If the training accuracy is high but validation accuracy is much lower, there is overfitting
elif accuracy_train > 0.9 and accuracy_val < 0.85:
    print("The model has high variance (overfitting). Recommendation: regularize the model or reduce complexity.")

# If the accuracies are close and high, the model is well-fitted
else:
    print("The model is well-fitted (good fit).")

# Classification report
class_report = classification_report(y_val, y_pred)
print("\nClassification Report:")
print(class_report)

# Calculate ROC-AUC for each class
roc_auc = roc_auc_score(y_val, model.predict_proba(X_val), multi_class='ovr')
print("ROC-AUC:", roc_auc)

# Perform cross-validation with 8 folds
cv = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
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