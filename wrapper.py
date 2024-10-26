# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Load dataset
data = pd.read_csv('Task_brainbeamy/DataScience_Task_1_Shabarish_B_L/Code/task1/breast_cancer/breast-cancer.csv')
data.drop(['id'], axis=1, inplace=True)
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

# Features and target
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

### Part 1: Recursive Feature Elimination (RFE) + SVM ###
# SVM for Recursive Feature Elimination
model_svm = SVC(kernel='linear', probability=False)  # No need for probability=True as we're not using ROC here
rfe = RFE(model_svm, n_features_to_select=10)  # Adjust number of features as needed
fit = rfe.fit(X, y)

# Selecting the top features using RFE
selected_features_rfe = X.columns[fit.support_]

# Split the data for RFE + SVM
X_train_rfe, X_test_rfe, y_train, y_test = train_test_split(X[selected_features_rfe], y, test_size=0.2, random_state=42)

# Train SVM with selected features
model_svm.fit(X_train_rfe, y_train)
y_pred_rfe = model_svm.predict(X_test_rfe)

# Evaluate RFE + SVM
accuracy_rfe_svm = accuracy_score(y_test, y_pred_rfe)
print(f'Accuracy with Recursive Feature Elimination + SVM: {accuracy_rfe_svm:.4f}')

### Part 2: Lasso Regression Feature Selection + XGBoost ###
# Standardize features for Lasso regression
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Lasso Regression with Cross-Validation for feature selection
lasso = LassoCV(cv=5, random_state=42, max_iter=10000, tol=0.01)  # Increase iterations and set tolerance
lasso.fit(X_scaled, y)

# Selecting the top features using Lasso (non-zero coefficients)
selected_features_lasso = X.columns[lasso.coef_ != 0]

# Split the data for Lasso + XGBoost
X_train_lasso, X_test_lasso, y_train, y_test = train_test_split(X[selected_features_lasso], y, test_size=0.2, random_state=42)

# Train XGBoost with Lasso-selected features
model_xgb = XGBClassifier(eval_metric='logloss', random_state=42)
model_xgb.fit(X_train_lasso, y_train)
y_pred_lasso = model_xgb.predict(X_test_lasso)

# Evaluate Lasso + XGBoost
accuracy_lasso_xgb = accuracy_score(y_test, y_pred_lasso)
print(f'Accuracy with Lasso Regression Feature Selection + XGBoost: {accuracy_lasso_xgb:.4f}')

### Heatmap Plotting for Confusion Matrices ###

# Function to plot confusion matrix heatmap
def plot_confusion_matrix_heatmap(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix Heatmap for {model_name}')
    plt.tight_layout()
    plt.show()

### Part 1: Heatmap for RFE + SVM ###
plot_confusion_matrix_heatmap(y_test, y_pred_rfe, 'RFE + SVM')

### Part 2: Heatmap for Lasso + XGBoost ###
plot_confusion_matrix_heatmap(y_test, y_pred_lasso, 'Lasso + XGBoost')
