import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
data = pd.read_csv('cdd.csv')

# Split the dataset
X = data.drop('Class', axis=1)
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply Random Forest Model
random_forest_model = RandomForestClassifier(random_state=42)
random_forest_model.fit(X_train, y_train)
y_pred_rf = random_forest_model.predict(X_test)

print("\nRandom Forest Model:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.2f}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_rf)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred_rf)}")
print("-" * 50)

# Apply Logistic Regression Model
logistic_regression_model = LogisticRegression(random_state=42, solver='liblinear')
logistic_regression_model.fit(X_train, y_train)
y_pred_lr = logistic_regression_model.predict(X_test)

print("\nLogistic Regression Model:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.2f}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_lr)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred_lr)}")
print("-" * 50)

# Apply Support Vector Machine (SVM) Model
svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

print("\nSupport Vector Machine (SVM) Model:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_svm):.2f}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_svm)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred_svm)}")
print("-" * 50)

# Apply K-Nearest Neighbors (KNN) Model
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

print("\nK-Nearest Neighbors (KNN) Model:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_knn):.2f}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_knn)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred_knn)}")
print("-" * 50)

# Taking input from the user
print("\nEnter values for the features to predict the class:")
feature_names = X.columns.tolist()
user_input = []

for feature in feature_names:
    value = float(input(f"Enter value for {feature}: "))
    user_input.append(value)

# Standardize user input
user_input = np.array(user_input).reshape(1, -1)
user_input = scaler.transform(user_input)    # Apply the same scaling as training

# Predicting using the trained model
predicted_class = random_forest_model.predict(user_input)
print(f"\nThe predicted class is: {predicted_class[0]} (Fraud Detected)" if predicted_class[0] == 1 else "(Not Fraud)")
