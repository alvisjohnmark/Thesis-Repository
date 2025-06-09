import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import itertools
import time

# Load datasets
df_train = pd.read_csv('/content/drive/MyDrive/thesis/train_feature_dominant_75.csv')
df_test = pd.read_csv('/content/drive/MyDrive/thesis/test_feature.csv')

# Preprocess datasets
def preprocess_data(df):
    df_cleaned = df.rename(columns=lambda col: col.rsplit('_', 1)[0])
    df_cleaned = df_cleaned.iloc[:, :-1].groupby(axis=1, level=0).mean()
    return df_cleaned

X_train = preprocess_data(df_train)
y_train = df_train.iloc[:, -1].astype(int)

X_test = preprocess_data(df_test)
y_test = df_test.iloc[:, -1].astype(int)

# Select best features
best_features = ['Chroma', 'MFCC', 'Spectral Contrast', 'Spectral Flatness', 'ZCR']
X_train = X_train[best_features]
X_test = X_test[best_features]

# Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train models
svm_model = SVC(kernel="rbf", C=10, gamma="scale", random_state=42)
svm_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)
rf_model.fit(X_train, y_train)

dt_model = DecisionTreeClassifier(max_depth=10, min_samples_split=5, random_state=42)
dt_model.fit(X_train, y_train)

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Brute-force feature selection
feature_columns = X_train.columns
best_results = {
    "SVM": {"accuracy": 0, "features": None},
    "Random Forest": {"accuracy": 0, "features": None},
    "Decision Tree": {"accuracy": 0, "features": None},
    "Naive Bayes": {"accuracy": 0, "features": None}
}

total_combinations = sum(1 for _ in itertools.chain.from_iterable(
    itertools.combinations(feature_columns, r) for r in range(1, len(feature_columns) + 1)
))
print(f"Evaluating {total_combinations} feature subsets...\n")

start_time = time.time()
subset_counter = 0

for r in range(1, len(feature_columns) + 1):
    for feature_subset in itertools.combinations(feature_columns, r):
        subset_counter += 1
        print(f"Testing subset {subset_counter}/{total_combinations}: {feature_subset}")

        X_subset = X_train[list(feature_subset)]

        X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
            X_subset, y_train, test_size=0.2, random_state=42, stratify=y_train
        )

        # Evaluate SVM
        svm_model = SVC(kernel="rbf", C=10, gamma="scale", random_state=42)
        svm_model.fit(X_train_split, y_train_split)
        svm_preds = svm_model.predict(X_test_split)
        svm_acc = accuracy_score(y_test_split, svm_preds)
        if svm_acc > best_results["SVM"]["accuracy"]:
            best_results["SVM"] = {"accuracy": svm_acc, "features": feature_subset}

        # Evaluate Random Forest
        rf_model = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)
        rf_model.fit(X_train_split, y_train_split)
        rf_preds = rf_model.predict(X_test_split)
        rf_acc = accuracy_score(y_test_split, rf_preds)
        if rf_acc > best_results["Random Forest"]["accuracy"]:
            best_results["Random Forest"] = {"accuracy": rf_acc, "features": feature_subset}

        # Evaluate Decision Tree
        dt_model = DecisionTreeClassifier(max_depth=10, min_samples_split=5, random_state=42)
        dt_model.fit(X_train_split, y_train_split)
        dt_preds = dt_model.predict(X_test_split)
        dt_acc = accuracy_score(y_test_split, dt_preds)
        if dt_acc > best_results["Decision Tree"]["accuracy"]:
            best_results["Decision Tree"] = {"accuracy": dt_acc, "features": feature_subset}

        # Evaluate Naive Bayes
        nb_model = GaussianNB()
        nb_model.fit(X_train_split, y_train_split)
        nb_preds = nb_model.predict(X_test_split)
        nb_acc = accuracy_score(y_test_split, nb_preds)
        if nb_acc > best_results["Naive Bayes"]["accuracy"]:
            best_results["Naive Bayes"] = {"accuracy": nb_acc, "features": feature_subset}

end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nCompleted in {elapsed_time:.2f} seconds.\n")

print("\nBest Results:")
for model in best_results:
    print(f"{model} Best Accuracy: {best_results[model]['accuracy']:.4f}")
    print(f"   Features used: {best_results[model]['features']}\n")