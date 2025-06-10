import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import KFold, cross_val_score
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

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


#SVM

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



#DECISION TREE
X = df.rename(columns=lambda col: col.rsplit('_', 1)[0])
X = X.iloc[:, :-1].groupby(axis=1, level=0).mean()
y = df.iloc[:, -1].astype(int)

best_features = ['MFCC', 'Spectral Centroid', 'Tempo', 'ZCR']
X = X[best_features]

scaler = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

dt_model = DecisionTreeClassifier(max_depth=10, min_samples_split=5, random_state=42)
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)
dt_acc = accuracy_score(y_test, dt_preds)
print(f"Decision Tree Accuracy: {dt_acc:.2f}")


# RANDOM FOREST
X = df.rename(columns=lambda col: col.rsplit('_', 1)[0])
X = X.iloc[:, :-1].groupby(axis=1, level=0).mean()
y = df.iloc[:, -1].astype(int)

# best_features = ['Chroma', 'RMS', 'Spectral Centroid', 'Tempo', 'ZCR']
best_features = ['Chroma', 'MFCC', 'Mel-Frequency', 'RMS', 'Spectral Centroid', 'Spectral Rolloff', 'Tempo', 'ZCR']
X = X[best_features]


scaler = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

rf_model = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds)
print(f"Random Forest  Accuracy: {rf_acc:.2f}")



# Compare model accuracies
model_accuracies = {
    "SVM": svm_acc,
    "Random Forest": rf_acc,
    "Decision Tree": dt_acc,
}

print("\nModel Accuracy Comparison:")
for model, acc in model_accuracies.items():
    print(f"{model}: {acc:.2f}")