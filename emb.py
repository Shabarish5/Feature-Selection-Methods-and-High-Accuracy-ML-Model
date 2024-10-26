# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv('Task_brainbeamy/DataScience_Task_1_Shabarish_B_L/Code/task1/breast_cancer/breast-cancer.csv')  # Include the correct path to your dataset
data.drop(['id'], axis=1, inplace=True)
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

# Features and target
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

### Part 1: Lasso Feature Selection + Random Forest ###
# Lasso Feature Selection
lasso = LassoCV(cv=5)
lasso.fit(X, y)

# Selecting non-zero coefficients
selected_features_lasso = X.columns[lasso.coef_ != 0]

# Split the data for Gradient Boosting
X_train_lasso, X_test_lasso, y_train, y_test = train_test_split(X[selected_features_lasso], y, test_size=0.2, random_state=42)

# Train Random Forest Classifier with Lasso-selected features
model_rf = RandomForestClassifier(random_state=42)
model_rf.fit(X_train_lasso, y_train)

# Evaluate Random Forest
y_pred_rf = model_rf.predict(X_test_lasso)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f'Accuracy with Lasso Feature Selection + Random Forest: {accuracy_rf:.4f}')

### Part 2: Variance Threshold Feature Selection + KNN ###
# Apply Variance Threshold to select features
vt = VarianceThreshold(threshold=0.01)  # Adjust threshold as necessary
X_vt = vt.fit_transform(X)

# Get the selected features after Variance Threshold
selected_features_vt = X.columns[vt.get_support()]

# Split the data for Variance Threshold + KNN
X_train_vt, X_test_vt, y_train, y_test = train_test_split(X_vt, y, test_size=0.2, random_state=42)

# Train KNN classifier with Variance Threshold-selected features
model_knn = KNeighborsClassifier(n_neighbors=5)  # Default k = 5
model_knn.fit(X_train_vt, y_train)
y_pred_knn = model_knn.predict(X_test_vt)

# Evaluate KNN
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f'Accuracy with Variance Threshold + KNN: {accuracy_knn:.4f}')

### Correlation Heatmap for Selected Features ###

# Correlation map for Lasso-selected features
plt.figure(figsize=(10, 8))
corr_lasso = data[selected_features_lasso].corr()
sns.heatmap(corr_lasso, annot=True, cmap="coolwarm", fmt=".2f")
plt.title('Correlation Heatmap for Lasso-Selected Features')
plt.tight_layout()
plt.show()

# Correlation map for Variance Threshold-selected features
plt.figure(figsize=(10, 8))
corr_vt = data[selected_features_vt].corr()
sns.heatmap(corr_vt, annot=True, cmap="coolwarm", fmt=".2f")
plt.title('Correlation Heatmap for Variance Threshold-Selected Features')
plt.tight_layout()
plt.show()
