import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import KFold, cross_val_score

# Evaluate models
def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.2f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))
    print("\nClassification Report:")
    print(classification_report(y_test, preds))
    return acc

# Evaluate SVM
print("SVM Performance:")
svm_acc = evaluate_model(svm_model, X_test, y_test)

# Evaluate Random Forest
print("Random Forest Performance:")
rf_acc = evaluate_model(rf_model, X_test, y_test)

# Evaluate Decision Tree
print("Decision Tree Performance:")
dt_acc = evaluate_model(dt_model, X_test, y_test)

# Evaluate Naive Bayes
print("Naive Bayes Performance:")
nb_acc = evaluate_model(nb_model, X_test, y_test)

# Cross-validation
def cross_validate_model(model, X, y, num_folds=10):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    results = cross_val_score(model, X, y, cv=kf)
    print("Cross-Validation Results:")
    for i, result in enumerate(results, 1):
        print(f"  Fold {i}: {result * 100:.2f}%")
    print(f"Mean Accuracy: {results.mean() * 100:.2f}%")

# Cross-validation for SVM
print("\nSVM Cross-Validation:")
cross_validate_model(svm_model, X_train, y_train)

# Cross-validation for Random Forest
print("\nRandom Forest Cross-Validation:")
cross_validate_model(rf_model, X_train, y_train)

# Cross-validation for Decision Tree
print("\nDecision Tree Cross-Validation:")
cross_validate_model(dt_model, X_train, y_train)

# Cross-validation for Naive Bayes
print("\nNaive Bayes Cross-Validation:")
cross_validate_model(nb_model, X_train, y_train)

# Compare model accuracies
model_accuracies = {
    "SVM": svm_acc,
    "Random Forest": rf_acc,
    "Decision Tree": dt_acc,
    "Naive Bayes": nb_acc
}

print("\nModel Accuracy Comparison:")
for model, acc in model_accuracies.items():
    print(f"{model}: {acc:.2f}")