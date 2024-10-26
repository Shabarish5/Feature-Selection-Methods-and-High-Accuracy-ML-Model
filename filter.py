# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, chi2
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load dataset
data = pd.read_csv('Task_brainbeamy/DataScience_Task_1_Shabarish_B_L/Code/task1/breast_cancer/breast-cancer.csv')

# Drop ID and irrelevant columns (adjust based on your dataset)
data.drop(['id'], axis=1, inplace=True)

# Convert categorical target to numerical if needed
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

# Correlation matrix visualization
correlation_matrix = data.corr()
plt.figure(figsize=(16, 14))  # Increase figure size for better readability
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', annot_kws={"size": 8})
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for clarity
plt.yticks(rotation=0)               # Rotate y-axis labels to horizontal
plt.title('Correlation Matrix of Breast Cancer Dataset', fontsize=15)
plt.tight_layout()  # Adjust layout to ensure nothing is cut off
plt.show()

### Part 1: Correlation-Based Filter + Logistic Regression ###
# Selecting highly correlated features (e.g., above 0.3 with the target)
correlation_threshold = 0.3
correlated_features = correlation_matrix['diagnosis'][abs(correlation_matrix['diagnosis']) > correlation_threshold].index
selected_features_corr = correlated_features.drop('diagnosis')

# Split the data for Correlation-Based Filter
X_corr = data[selected_features_corr]
y = data['diagnosis']
X_train_corr, X_test_corr, y_train_corr, y_test_corr = train_test_split(X_corr, y, test_size=0.2, random_state=42)

# Logistic Regression with Correlation-Based Features
logistic_model_corr = LogisticRegression(max_iter=10000)
logistic_model_corr.fit(X_train_corr, y_train_corr)
y_pred_logistic_corr = logistic_model_corr.predict(X_test_corr)

# Evaluate Logistic Regression
accuracy_logistic_corr = accuracy_score(y_test_corr, y_pred_logistic_corr)
print(f'Accuracy with Correlation-Based Filter + Logistic Regression: {accuracy_logistic_corr:.4f}')

### Part 2: Chi-Square Test + Decision Tree ###
# Features and target for Chi-Square Test
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

# Normalize feature data for Chi-Square Test
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Apply Chi-Square Test
chi2_selector = SelectKBest(chi2, k=10)  # Select top 10 features based on Chi-Square Test
X_chi2 = chi2_selector.fit_transform(X_scaled, y)
selected_features_chi2 = X.columns[chi2_selector.get_support()]

# Split the data for Chi-Square selected features
X_train_chi2, X_test_chi2, y_train_chi2, y_test_chi2 = train_test_split(X[selected_features_chi2], y, test_size=0.2, random_state=42)

# Decision Tree Classifier with Chi-Square selected features
decision_tree_model_chi2 = DecisionTreeClassifier(random_state=42)
decision_tree_model_chi2.fit(X_train_chi2, y_train_chi2)
y_pred_tree_chi2 = decision_tree_model_chi2.predict(X_test_chi2)

# Evaluate Decision Tree
accuracy_tree_chi2 = accuracy_score(y_test_chi2, y_pred_tree_chi2)
print(f'Accuracy with Chi-Square Test + Decision Tree: {accuracy_tree_chi2:.4f}')

# Visualize the Decision Tree
plt.figure(figsize=(24, 16), dpi=150)  # Increase figure size and DPI for better clarity
plot_tree(decision_tree_model_chi2, 
          feature_names=selected_features_chi2, 
          class_names=['Benign', 'Malignant'], 
          filled=True, 
          rounded=True, 
          fontsize=3,  # Reduce font size
          max_depth=8)  # Optionally limit tree depth
plt.title('Decision Tree Visualization (Chi-Square Test Selected Features)', fontsize=15)
plt.tight_layout()  # Adjust layout for better spacing
plt.show()
