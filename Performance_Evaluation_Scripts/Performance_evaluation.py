import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import KFold, cross_val_score
from scipy.stats import zscore

df = pd.read_csv('/feature_extraction_output.csv')

def replace_outliers_with_mean_zscore(df, threshold=3):
    df_cleaned = df.copy()

    for col in df_cleaned.columns:
        # Calculate Z-scores
        z_scores = np.abs(zscore(df_cleaned[col]))

        # Identify outliers
        outliers = z_scores > threshold

        # Calculate mean without outliers
        mean_value = df_cleaned.loc[~outliers, col].mean()

        # Replace outliers with mean
        df_cleaned.loc[outliers, col] = mean_value

    return df_cleaned

# Apply function
df_cleaned = replace_outliers_with_mean_zscore(df)
# print(df_cleaned.shape)

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

X = df_cleaned.rename(columns=lambda col: col.rsplit('_', 1)[0])
X = X.iloc[:, :-1].groupby(axis=1, level=0).mean()
y = df.iloc[:, -1].astype(int)


best_features = ['Chroma', 'MFCC', 'Spectral Contrast', 'Spectral Flatness', 'ZCR']


X = X[best_features]

scaler = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# X_train = replace_outliers_with_mean_zscore(X_train)
# X_test = replace_outliers_with_mean_zscore(X_test)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm_model = SVC(kernel="rbf", C=10, gamma="scale", random_state=42)
svm_model.fit(X_train, y_train)
svm_preds = svm_model.predict(X_test)
svm_acc = accuracy_score(y_test, svm_preds)
print(f"SVM Accuracy: {svm_acc:.2f}")

# Evaluate SVM
print("SVM Performance:")
svm_acc = evaluate_model(svm_model, X_test, y_test)

# Evaluate Random Forest
print("Random Forest Performance:")
rf_acc = evaluate_model(rf_model, X_test, y_test)

# Evaluate Decision Tree
print("Decision Tree Performance:")
dt_acc = evaluate_model(dt_model, X_test, y_test)


# Compare model accuracies
model_accuracies = {
    "SVM": svm_acc,
    "Random Forest": rf_acc,
    "Decision Tree": dt_acc,
}

print("\nModel Accuracy Comparison:")
for model, acc in model_accuracies.items():
    print(f"{model}: {acc:.2f}")