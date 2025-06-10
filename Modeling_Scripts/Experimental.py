# -*- coding: utf-8 -*-
"""experiment_features
"""

# Importing required libraries
# Keras
# sklearn
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Other
import librosa
import librosa.display
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import pandas as pd
import seaborn as sns
import glob
import os
import pickle
import IPython.display as ipd  # To play sound in the notebook
from google.colab import drive
from sklearn.preprocessing import MinMaxScaler, StandardScaler
drive.mount('/content/drive/', force_remount=True)

#experiment
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import itertools
from google.colab import files
import time
from sklearn.svm import SVC
from sklearn.feature_selection import SequentialFeatureSelector
from joblib import Memory
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

file = '../audio_features_25_soxrHQ_44_all_stacked_189.csv'
df = pd.read_csv(file)
df.head()

audio_features = [
    (range(0, 40), 'MFCC'),
    (range(40, 168), 'Mel-Frequency'),
    (range(168, 180), 'Chroma'),
    (range(180, 181), 'Spectral Rolloff'),
    (range(181, 182), 'Spectral Bandwidth'),
    (range(182, 183), 'Spectral Contrast'),
    (range(183, 184), 'Spectral Flatness'),
    (range(184, 185), 'Spectral Centroid'),
    (range(185, 186), 'RMS'),
    (range(186, 187), 'Tempo'),
    (range(187, 188), 'ZCR'),
    (range(188, 189), 'Label'),
]
col_names = []
for feature_range, feature_name in audio_features:
    col_names.extend([f"{feature_name}_{i}" for i in feature_range])

df.columns = col_names
df

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

X = df.rename(columns=lambda col: col.rsplit('_', 1)[0])
X = X.iloc[:, :-1].groupby(axis=1, level=0).mean()
y = df.iloc[:, -1].astype(int)
X
# scaler = StandardScaler()
# X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
# X_scaled

"""# **Enumerate all possible combinations and find best accuracy**

### Cleaned/removed outliers
"""

X = df_cleaned.rename(columns=lambda col: col.rsplit('_', 1)[0])
X = X.iloc[:, :-1].groupby(axis=1, level=0).mean()

columns = X.columns


scaler = StandardScaler()
y = df_cleaned.iloc[:, -1].astype(int)
print(X.shape)


best_results = {
    "SVM": {"accuracy": 0, "features": None},
    "Random Forest": {"accuracy": 0, "features": None},
    "Decision Tree": {"accuracy": 0, "features": None},
    "Naives Bayes": {"accuracy": 0, "features": None}
}

print(f"Evaluating feature subsets...\n")

start_time = time.time()
subset_counter = 0

for r in range(1, len(columns) + 1):
    for feature_subset in itertools.combinations(columns, r):
        subset_counter += 1
        print(f"Testing subset {subset_counter}: {feature_subset}")



        X_subset = X[list(feature_subset)]

        X_train, X_test, y_train, y_test = train_test_split(X_subset, y, test_size=0.2, random_state=42, stratify=y)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        svm_model = SVC(kernel="rbf", C=10, gamma="scale", random_state=42)
        svm_model.fit(X_train, y_train)
        svm_preds = svm_model.predict(X_test)
        svm_acc = accuracy_score(y_test, svm_preds)
        print("SVM: ", svm_acc)

        if svm_acc > best_results["SVM"]["accuracy"]:
            best_results["SVM"] = {"accuracy": svm_acc, "features": feature_subset}


        rf_model = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_preds = rf_model.predict(X_test)
        rf_acc = accuracy_score(y_test, rf_preds)
        print("REgression Tree: ", rf_acc)

        if rf_acc > best_results["Random Forest"]["accuracy"]:
            best_results["Random Forest"] = {"accuracy": rf_acc, "features": feature_subset}

        dt_model = DecisionTreeClassifier(max_depth=10, min_samples_split=5, random_state=42)
        dt_model.fit(X_train, y_train)
        dt_preds = dt_model.predict(X_test)
        dt_acc = accuracy_score(y_test, dt_preds)
        print("Decision Tree: ", dt_acc)

        if dt_acc > best_results["Decision Tree"]["accuracy"]:
            best_results["Decision Tree"] = {"accuracy": dt_acc, "features": feature_subset}

        nb_model = GaussianNB()
        nb_model.fit(X_train, y_train)
        nb_pred = nb_model.predict(X_test)
        nb_acc = accuracy_score(y_test, nb_pred)
        if nb_acc > best_results["Naives Bayes"]["accuracy"]:
          best_results["Naives Bayes"] = {"accuracy": nb_acc, "features": feature_subset}
        print("Naives Bayes: ", nb_acc)


end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nCompleted in {elapsed_time:.2f} seconds.\n")


print("\nBest Results:")
for model in best_results:
    print(f"{model} Best Accuracy: {best_results[model]['accuracy']:.4f}")
    print(f"   Features used: {best_results[model]['features']}\n")

import matplotlib.pyplot as plt
models = ["SVM", "RF", "DT", "NB"]
accuracies = [92, 88, 77, 85]  # Accuracy percentages

# Create the horizontal bar graph
plt.figure(figsize=(8, 3))
plt.barh(models, accuracies, color=["blue", "green", "red", "orange"], height=.8)

# Labels and title
plt.xlabel("Accuracy (%)")
plt.ylabel("Models")
plt.title("Model Best Accuracy Comparison")

# Show values on bars
for index, value in enumerate(accuracies):
    plt.text(value + 1, index, f"{value}%", va='center')

plt.xlim(0, 100)  # Set x-axis limit from 0 to 100%
plt.grid(axis="x", linestyle="--", alpha=0.6)

plt.show()

"""# **Plots**"""

X = df_cleaned.rename(columns=lambda col: col.rsplit('_', 1)[0])
X = X.iloc[:, :-1].groupby(axis=1, level=0).mean()

columns = X.columns


scaler = StandardScaler()
y = df_cleaned.iloc[:, -1].astype(int)
print(X.shape)

accuracy_results = {
    "SVM": [],
    "Random Forest": [],
    "Decision Tree": [],
    "Naïve Bayes": []
}

best_results = {
    "SVM": {"accuracy": 0, "features": None},
    "Random Forest": {"accuracy": 0, "features": None},
    "Decision Tree": {"accuracy": 0, "features": None},
    "Naïve Bayes": {"accuracy": 0, "features": None}
}

print("Evaluating feature subsets...\n")
start_time = time.time()
subset_counter = 0

for r in range(1, len(columns) + 1):
    for feature_subset in itertools.combinations(columns, r):
        subset_counter += 1
        print(f"Testing subset {subset_counter}: {feature_subset}")

        X_subset = X[list(feature_subset)]

        X_train, X_test, y_train, y_test = train_test_split(X_subset, y, test_size=0.2, random_state=42, stratify=y)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        svm_model = SVC(kernel="rbf", C=10, gamma="scale", random_state=42)
        svm_model.fit(X_train, y_train)
        svm_preds = svm_model.predict(X_test)
        svm_acc = accuracy_score(y_test, svm_preds)
        accuracy_results["SVM"].append((svm_acc, len(feature_subset)))
        print("SVM: ", svm_acc)

        if svm_acc > best_results["SVM"]["accuracy"]:
            best_results["SVM"] = {"accuracy": svm_acc, "features": feature_subset}


        rf_model = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_preds = rf_model.predict(X_test)
        rf_acc = accuracy_score(y_test, rf_preds)
        accuracy_results["Random Forest"].append((rf_acc, len(feature_subset)))
        print("Random Forest: ", rf_acc)

        if rf_acc > best_results["Random Forest"]["accuracy"]:
            best_results["Random Forest"] = {"accuracy": rf_acc, "features": feature_subset}

        dt_model = DecisionTreeClassifier(max_depth=10, min_samples_split=5, random_state=42)
        dt_model.fit(X_train, y_train)
        dt_preds = dt_model.predict(X_test)
        dt_acc = accuracy_score(y_test, dt_preds)
        accuracy_results["Decision Tree"].append((dt_acc, len(feature_subset)))
        print("Decision Tree: ", dt_acc)

        if dt_acc > best_results["Decision Tree"]["accuracy"]:
            best_results["Decision Tree"] = {"accuracy": dt_acc, "features": feature_subset}

        nb_model = GaussianNB()
        nb_model.fit(X_train, y_train)
        nb_pred = nb_model.predict(X_test)
        nb_acc = accuracy_score(y_test, nb_pred)
        accuracy_results["Naïve Bayes"].append((nb_acc, len(feature_subset)))
        print("Naïve Bayes: ", nb_acc)

        if nb_acc > best_results["Naïve Bayes"]["accuracy"]:
          best_results["Naïve Bayes"] = {"accuracy": nb_acc, "features": feature_subset}


end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nCompleted in {elapsed_time:.2f} seconds.\n")


print("\nBest Results:")
for model in best_results:
    print(f"{model} Best Accuracy: {best_results[model]['accuracy']:.4f}")
    print(f"   Features used: {best_results[model]['features']}\n")

print(best_results)

print(accuracy_results)

bin_size = 5  # for visualization, less clutter
colors = {"SVM": "blue", "Random Forest": "green", "Decision Tree": "red", "Naïve Bayes": "orange"}

for model, accuracies in accuracy_results.items():
    plt.figure(figsize=(15, 5))

    bins = np.arange(0, len(accuracies), bin_size)
    binned_accuracies = [np.mean(accuracies[i:i + bin_size]) for i in bins]

    bin_centers = bins + bin_size // 2

    plt.plot(bin_centers, binned_accuracies, marker='o', linestyle='-', color=colors[model], label=model)

    plt.xlabel("Feature Subsets")
    plt.ylabel("Accuracy Score")
    plt.title(f"Model Accuracy for {model}")
    plt.ylim(0.1, 0.9)
    plt.legend()
    plt.grid(True)

    plt.show()

import matplotlib.pyplot as plt

colors = {
    "SVM": "lightblue",
    "Random Forest": "lightblue",
    "Decision Tree": "lightblue",
    "Naïve Bayes": "lightblue"
}

plt.figure(figsize=(12, 6))

all_feature_counts = set()

for model, results in accuracy_results.items():
    best_per_count = {}
    if model == "Naïve Bayes":
      continue
    for acc, feature_count in results:
        all_feature_counts.add(feature_count)
        if feature_count not in best_per_count:
            best_per_count[feature_count] = acc
        else:
            best_per_count[feature_count] = max(best_per_count[feature_count], acc)

    # Sort values for plotting
    sorted_counts = sorted(best_per_count.keys())
    sorted_accuracies = [best_per_count[count] for count in sorted_counts]

    # Plot each model
    plt.plot(sorted_counts, sorted_accuracies, marker='o', linestyle='-', color=colors.get(model, "black"), label=model)

# X-axis ticks: show all feature counts that appeared in any model
plt.xticks(sorted(sorted(all_feature_counts)))

plt.xlabel("Number of Features")
plt.ylabel("Best Accuracy")
plt.title("Best Accuracy per Feature Count for Each Model")
plt.ylim(0.1, 1.0)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

colors = {
    "SVM": "blue",
    "Random Forest": "green",
    "Decision Tree": "red",
}

plt.figure(figsize=(12, 6))

all_feature_counts = set()

for model, results in accuracy_results.items():
    # Group accuracies by feature count
    if model == "Naïve Bayes":
      continue
    grouped = defaultdict(list)
    for acc, feature_count in results:
        grouped[feature_count].append(acc)
        all_feature_counts.add(feature_count)

    # Compute mean per feature count
    sorted_counts = sorted(grouped.keys())
    mean_accuracies = [np.mean(grouped[count]) for count in sorted_counts]

    # Plot each model
    plt.plot(sorted_counts, mean_accuracies, marker='o', linestyle='-', color=colors.get(model, "black"), label=model)

# X-axis settings
plt.xticks(sorted(all_feature_counts))

plt.xlabel("Number of Features")
plt.ylabel("Mean Accuracy")
plt.title("Mean Accuracy per Feature Count (Per Model)")
plt.ylim(0.1, 1.0)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

data = {
    "SVM": {"Raw": 0.77, "Aggregated": 0.92},
    "Random Forest": {"Raw": 0.77, "Aggregated": 0.88},
    "Decision Tree": {"Raw": 0.81, "Aggregated": 0.77},
}

models = list(data.keys())
bar_width = 0.35
bar_gap = 0.1
index = np.arange(len(models))

# Scale factor to reduce bar length visually
scale = 0.8

# Scaled values
raw_scores = [data[model]["Raw"] * scale for model in models]
agg_scores = [data[model]["Aggregated"] * scale for model in models]

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

# Plot bars with spacing
ax.barh(index - (bar_width / 2 + bar_gap / 2), raw_scores, bar_width, label='LL features', color='#0074b3')
ax.barh(index + (bar_width / 2 + bar_gap / 2), agg_scores, bar_width, label='LL+Mean Features', color='#ff7f0e')

# Set original value ticks, not scaled
x_ticks = np.arange(0.0, 1.1, 0.1)
ax.set_xticks([tick * scale for tick in x_ticks])
ax.set_xticklabels([f"{tick:.1f}" for tick in x_ticks])

# Labels and formatting
ax.set(yticks=index, yticklabels=models)
ax.set_xlabel('Accuracy')
ax.set_title('Model Performance: Low-level vs Low-level and Mean Features')
ax.legend()
ax.grid(axis='x', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

"""# **MODEL TUNE**

# Best features from each model

### SVM
"""

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

print("Confusion Matrix:")
print(confusion_matrix(y_test, svm_preds))

print("\nClassification Report:")
print(classification_report(y_test, svm_preds))

num_folds = 10
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
cross_val_results = cross_val_score(svm_model, X, y, cv=kf)
print("Cross-Validation Results (Accuracy):")
for i, result in enumerate(cross_val_results, 1):
    print(f"  Fold {i}: {result * 100:.2f}%")

print(f'Mean Accuracy: {cross_val_results.mean()* 100:.2f}%')

"""### Decision Tree"""

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

print("Confusion Matrix:")
print(confusion_matrix(y_test, dt_preds))

print("\nClassification Report:")
print(classification_report(y_test, dt_preds))

num_folds = 10
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
cross_val_results = cross_val_score(dt_model, X, y, cv=kf)
print("Cross-Validation Results (Accuracy):")
for i, result in enumerate(cross_val_results, 1):
    print(f"  Fold {i}: {result * 100:.2f}%")

print(f'Mean Accuracy: {cross_val_results.mean()* 100:.2f}%')

"""### Random Forests"""

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

print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_preds))

print("\nClassification Report:")
print(classification_report(y_test, rf_preds))

num_folds = 10
kf = KFold(n_splits=num_folds, shuffle=True, random_state=12)
cross_val_results = cross_val_score(rf_model, X, y, cv=kf)
print("Cross-Validation Results (Accuracy):")
for i, result in enumerate(cross_val_results, 1):
    print(f"  Fold {i}: {result * 100:.2f}%")

print(f'Mean Accuracy: {cross_val_results.mean()* 100:.2f}%')

from collections import Counter

best_features = ['Chroma', 'MFCC', 'Spectral Contrast', 'Spectral Flatness', 'Spectral Rolloff', 'ZCR']
feature_counts = Counter()

for col in df.columns:
    for feature in best_features:
        if feature in col:
            feature_counts[feature] += 1

for feature, count in feature_counts.items():
    print(f"{feature}: {count} columns")

"""# Dynamic labelling"""

df3 = pd.read_csv("/content/drive/MyDrive/thesis/dataset/APT_segments.csv")
df3.drop(df3.columns[0], axis=1, inplace=True)
df3.shape

audio_features = [
    (range(0, 40), 'MFCC'),
    (range(40, 168), 'Mel-Frequency'),
    (range(168, 180), 'Chroma'),
    (range(180, 181), 'Spectral Rolloff'),
    (range(181, 182), 'Spectral Bandwidth'),
    (range(182, 183), 'Spectral Contrast'),
    (range(183, 184), 'Spectral Flatness'),
    (range(184, 185), 'Spectral Centroid'),
    (range(185, 186), 'RMS'),
    (range(186, 187), 'Tempo'),
    (range(187, 188), 'ZCR'),
]
col_names = []
for feature_range, feature_name in audio_features:
    col_names.extend([f"{feature_name}_{i}" for i in feature_range])

df3.columns = col_names
df3

df3_cleaned = replace_outliers_with_mean_zscore(df3)

X = df3_cleaned.rename(columns=lambda col: col.rsplit('_', 1)[0])
X = X.groupby(axis=1, level=0).mean()
X

best_features = ['Chroma', 'MFCC', 'Spectral Contrast', 'Spectral Flatness', 'ZCR']

X_best = X[best_features]
X_best

X_scaled = scaler.fit_transform(X_best)

y_new_pred = svm_model.predict(X_scaled)
# svm_acc = accuracy_score(y_test, y_pred)
print(f"Random Forest: {svm_acc:.2f}")

X_best["Emotion Profile"] = [label for label in y_new_pred]
X_best

X_best.groupby("Emotion Profile").count()

# Simulated emotion labels over time (0 = Happy, 1 = Sad, 2 = Angry, 3 = Calm)
emotion_changes_100 = X_best['Emotion Profile']
seconds = [x * 3 for x in range(1, 56)]

# Plot the emotions over time
plt.figure(figsize=(12, 5))
plt.plot(seconds, emotion_changes_100, label='Emotion Change', color='lightblue', linestyle='solid', marker='o', markersize=5)

# Labels and title
plt.xlabel('Time (seconds)')
plt.ylabel('Emotion Category')
plt.title('\'APT\' Emotion Profile Over Time (100%)')
plt.yticks([0, 1, 2, 3], ['Happy', 'Angry', 'Sad', 'Calm'])
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

path = '/content/drive/MyDrive/thesis/dataset/recognized_APT_segments.csv'
with open(path, 'w', encoding = 'utf-8-sig') as f:
  X_best.to_csv(f)

import matplotlib.pyplot as plt
emotions = ['Happy', 'Angry', 'Sad', 'Calm']
counts = [18, 16, 8, 13]  # Example values, adjust as needed


# Create the horizontal bar graph
plt.figure(figsize=(8, 3))
plt.barh(emotions, counts,  height=.8)

# Labels and title
plt.xlabel("Frequency")
plt.ylabel("Emotions")
plt.title("Emotion Distribution (100%)")

plt.show()

data = '/content/drive/MyDrive/thesis/audio_dataset.csv'
# data = {
#     'path': [path_audio + str(i) + '.wav' for i in range(1, 21)],
#     'label': [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]
# }
data = pd.read_csv(data)
df = pd.DataFrame(data)

import numpy as np
import matplotlib.pyplot as plt
import math # Importing math module

rms_values = rms1[0]

# 1. Compute histogram data manually
counts, bin_edges = np.histogram(rms_values, bins=50)

# 2. Compute bin centers
bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

# 3. Sort bins by frequency (descending)
sorted_indices = np.argsort(counts)[::-1]
sorted_counts = counts[sorted_indices]
sorted_bin_centers = bin_centers[sorted_indices]
p = sorted_bin_centers[:math.floor(len(bin_centers) *.25 )]
print(np.mean(p))

# 4. Plot sorted histogram
plt.figure(figsize=(10, 5))
plt.bar(range(len(sorted_counts)), sorted_counts, width=1, color='orange')
plt.xticks(range(len(sorted_counts)), [f"{val:.3f}" for val in sorted_bin_centers], rotation=90)

plt.title('Histogram of RMS Values (Sorted by Frequency)')
plt.xlabel('RMS Bin (sorted)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Generate right-skewed data using exponential distribution
np.random.seed(0)
data = np.random.exponential(scale=1.0, size=3000)

# Get the min and max of the data
data_min, data_max = np.min(data), np.max(data)

# Divide the range into 4 equal-width intervals
equal_divisions = np.linspace(data_min, data_max, 5)

# Plot histogram
plt.figure(figsize=(10, 6))
counts, bins, patches = plt.hist(data, bins=50, color='salmon', edgecolor='black', alpha=0.75)

# Add vertical lines for 25%, 50%, 75%, 100% (equally spaced in range, not by data percentiles)
for q, label in zip(equal_divisions[1:], ['25%', '50%', '75%', '100%']):
    plt.axvline(q, color='green', linestyle='--', linewidth=2)
    plt.text(q, max(counts)*0.9, label, rotation=90, verticalalignment='center', color='green', fontsize=18)

# Labels and title
plt.title('Right-Skewed Histogram with Equal-Range Divisions')
plt.xlabel('Value')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

df_train = pd.read_csv('/content/drive/MyDrive/thesis/train_feature_dominant_75.csv')
df_test = pd.read_csv('/content/drive/MyDrive/thesis/test_feature.csv')

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


scaler = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(df_train, df_test, test_size=0.2, random_state=42, stratify=y)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm_model = SVC(kernel="rbf", C=10, gamma="scale", random_state=42)
svm_model.fit(X_train, y_train)
svm_preds = svm_model.predict(X_test)
svm_acc = accuracy_score(y_test, svm_preds)
print(f"SVM Accuracy: {svm_acc:.2f}")

print("Confusion Matrix:")
print(confusion_matrix(y_test, svm_preds))

print("\nClassification Report:")
print(classification_report(y_test, svm_preds))
print("SVM: ", svm_acc)

rf_model = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds)
print(f"RF Accuracy: {rf_acc:.2f}")

print("Confusion Matrix:")
print(confusion_matrix(y_test, svm_preds))

print("\nClassification Report:")
print(classification_report(y_test, svm_preds))
print("SVM: ", rf_acc)

dt_model = DecisionTreeClassifier(max_depth=10, min_samples_split=5, random_state=42)
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)
dt_acc = accuracy_score(y_test, dt_preds)
print(f"DT Accuracy: {dt_acc:.2f}")

print("Confusion Matrix:")
print(confusion_matrix(y_test, svm_preds))

print("\nClassification Report:")
print(classification_report(y_test, svm_preds))
print("SVM: ", dt_acc)

X_train = df_train.rename(columns=lambda col: col.rsplit('_', 1)[0])
X_train = X_train.iloc[:, :-1].groupby(axis=1, level=0).mean()
y_train = df_train.iloc[:, -1].astype(int)

X_test = df_test.rename(columns=lambda col: col.rsplit('_', 1)[0])
X_test = X_test.iloc[:, :-1].groupby(axis=1, level=0).mean()
y_test = df_test.iloc[:, -1].astype(int)

#25%
X_train = df_train.rename(columns=lambda col: col.rsplit('_', 1)[0])
X_train = X_train.iloc[:, :-1].groupby(axis=1, level=0).mean()
X_train = replace_outliers_with_mean_zscore(X_train)
y_train = df_train.iloc[:, -1].astype(int)

X_test = df_test.rename(columns=lambda col: col.rsplit('_', 1)[0])
X_test = X_test.iloc[:, :-1].groupby(axis=1, level=0).mean()
X_test = replace_outliers_with_mean_zscore(X_test)
y_test = df_test.iloc[:, -1].astype(int)

best_features = ['Chroma', 'MFCC', 'Spectral Contrast', 'Spectral Flatness', 'ZCR']

X_train = X_train[best_features]
X_test = X_test[best_features]

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm_model = SVC(kernel="rbf", C=10, gamma="scale", random_state=42)
svm_model.fit(X_train, y_train)
svm_preds = svm_model.predict(X_test)
svm_acc = accuracy_score(y_test, svm_preds)
print(f"SVM Accuracy: {svm_acc:.2f}")

print("Confusion Matrix:")
print(confusion_matrix(y_test, svm_preds))

print("\nClassification Report:")
print(classification_report(y_test, svm_preds))


df3 = pd.read_csv("/content/drive/MyDrive/thesis/dataset/APT_segments.csv")
df3.drop(df3.columns[0], axis=1, inplace=True)
df3.shape

audio_features = [
    (range(0, 40), 'MFCC'),
    (range(40, 168), 'Mel-Frequency'),
    (range(168, 180), 'Chroma'),
    (range(180, 181), 'Spectral Rolloff'),
    (range(181, 182), 'Spectral Bandwidth'),
    (range(182, 183), 'Spectral Contrast'),
    (range(183, 184), 'Spectral Flatness'),
    (range(184, 185), 'Spectral Centroid'),
    (range(185, 186), 'RMS'),
    (range(186, 187), 'Tempo'),
    (range(187, 188), 'ZCR'),
]
col_names = []
for feature_range, feature_name in audio_features:
    col_names.extend([f"{feature_name}_{i}" for i in feature_range])

df3.columns = col_names

best_features = ['Chroma', 'MFCC', 'Spectral Contrast', 'Spectral Flatness', 'ZCR']
df3_cleaned = replace_outliers_with_mean_zscore(df3)

X = df3_cleaned.rename(columns=lambda col: col.rsplit('_', 1)[0])
X = X.groupby(axis=1, level=0).mean()
X
X_best = X[best_features]
X_best

X_scaled = scaler.fit_transform(X_best)

y_new_pred = svm_model.predict(X_scaled)
# svm_acc = accuracy_score(y_test, y_pred)
print(f"Random Forest: {svm_acc:.2f}")

X_best["Emotion Profile"] = [label for label in y_new_pred]

X_best.groupby("Emotion Profile").count()

emotions = ['Happy', 'Angry', 'Sad', 'Calm']
counts = [23, 17, 7, 8]  # Example values, adjust as needed


# Create the horizontal bar graph
plt.figure(figsize=(8, 3))
plt.barh(emotions, counts,  height=.8)

# Labels and title
plt.xlabel("Frequency")
plt.ylabel("Emotions")
plt.title("Emotion Distribution (25%)")

plt.show()

# Simulated emotion labels over time (0 = Happy, 1 = Sad, 2 = Angry, 3 = Calm)
emotion_changes_25 = X_best['Emotion Profile']
seconds = [x * 3 for x in range(1, 56)]

# Plot the emotions over time
plt.figure(figsize=(12, 5))
plt.plot(seconds, emotion_changes_25, label='Emotion Change', color='lightblue', linestyle='solid', marker='o', markersize=5)

# Labels and title
plt.xlabel('Time (seconds)')
plt.ylabel('Emotion Category')
plt.title('\'APT\' Emotion Profile Over Time (25%)')
plt.yticks([0, 1, 2, 3], ['Happy', 'Angry', 'Sad', 'Calm'])
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

#50%
X_train = df_train.rename(columns=lambda col: col.rsplit('_', 1)[0])
X_train = X_train.iloc[:, :-1].groupby(axis=1, level=0).mean()
X_train = replace_outliers_with_mean_zscore(X_train)
y_train = df_train.iloc[:, -1].astype(int)

X_test = df_test.rename(columns=lambda col: col.rsplit('_', 1)[0])
X_test = X_test.iloc[:, :-1].groupby(axis=1, level=0).mean()
X_test = replace_outliers_with_mean_zscore(X_test)
y_test = df_test.iloc[:, -1].astype(int)

best_features = ['Chroma', 'MFCC', 'Spectral Contrast', 'Spectral Flatness', 'ZCR']

X_train = X_train[best_features]
X_test = X_test[best_features]

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm_model = SVC(kernel="rbf", C=10, gamma="scale", random_state=42)
svm_model.fit(X_train, y_train)
svm_preds = svm_model.predict(X_test)
svm_acc = accuracy_score(y_test, svm_preds)
print(f"SVM Accuracy: {svm_acc:.2f}")

print("Confusion Matrix:")
print(confusion_matrix(y_test, svm_preds))

print("\nClassification Report:")
print(classification_report(y_test, svm_preds))


y_new_pred = svm_model.predict(X_scaled)
# svm_acc = accuracy_score(y_test, y_pred)
print(f"Random Forest: {svm_acc:.2f}")

X_best["Emotion Profile"] = [label for label in y_new_pred]

X_best.groupby("Emotion Profile").count()

emotions = ['Happy', 'Angry', 'Sad', 'Calm']
counts = [21, 17, 6,11]  # Example values, adjust as needed


# Create the horizontal bar graph
plt.figure(figsize=(8, 3))
plt.barh(emotions, counts,  height=.8)

# Labels and title
plt.xlabel("Frequency")
plt.ylabel("Emotions")
plt.title("Emotion Distribution (50%)")

plt.show()

# Simulated emotion labels over time (0 = Happy, 1 = Sad, 2 = Angry, 3 = Calm)
emotion_changes_50 = X_best['Emotion Profile']
seconds = [x * 3 for x in range(1, 56)]

# Plot the emotions over time
plt.figure(figsize=(12, 5))
plt.plot(seconds, emotion_changes_50, label='Emotion Change', color='lightblue', linestyle='solid', marker='o', markersize=5)

# Labels and title
plt.xlabel('Time (seconds)')
plt.ylabel('Emotion Category')
plt.title('\'APT\' Emotion Profile Over Time (50%)')
plt.yticks([0, 1, 2, 3], ['Happy', 'Angry', 'Sad', 'Calm'])
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

#75%
X_train = df_train.rename(columns=lambda col: col.rsplit('_', 1)[0])
X_train = X_train.iloc[:, :-1].groupby(axis=1, level=0).mean()
y_train = df_train.iloc[:, -1].astype(int)

X_test = df_test.rename(columns=lambda col: col.rsplit('_', 1)[0])
X_test = X_test.iloc[:, :-1].groupby(axis=1, level=0).mean()
y_test = df_test.iloc[:, -1].astype(int)

best_features = ['Chroma', 'MFCC', 'Spectral Contrast', 'Spectral Flatness', 'ZCR']

X_train = X_train[best_features]
X_test = X_test[best_features]

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm_model = SVC(kernel="rbf", C=10, gamma="scale", random_state=42)
svm_model.fit(X_train, y_train)
svm_preds = svm_model.predict(X_test)
svm_acc = accuracy_score(y_test, svm_preds)
print(f"SVM Accuracy: {svm_acc:.2f}")

print("Confusion Matrix:")
print(confusion_matrix(y_test, svm_preds))

print("\nClassification Report:")
print(classification_report(y_test, svm_preds))


y_new_pred = svm_model.predict(X_scaled)
print(f"Random Forest: {svm_acc:.2f}")

X_best["Emotion Profile"] = [label for label in y_new_pred]

X_best.groupby("Emotion Profile").count()

emotions = ['Happy', 'Angry', 'Sad', 'Calm']
counts = [24, 15, 11, 5]  # Example values, adjust as needed


# Create the horizontal bar graph
plt.figure(figsize=(8, 3))
plt.barh(emotions, counts,  height=.8)

# Labels and title
plt.xlabel("Frequency")
plt.ylabel("Emotions")
plt.title("Emotion Distribution (75%)")

plt.show()

# Simulated emotion labels over time (0 = Happy, 1 = Sad, 2 = Angry, 3 = Calm)
emotion_changes_75 = X_best['Emotion Profile']
seconds = [x * 3 for x in range(1, 56)]

# Plot the emotions over time
plt.figure(figsize=(12, 5))
plt.plot(seconds, emotion_changes_75, label='Emotion Change', color='lightblue', linestyle='solid', marker='o', markersize=5)

# Labels and title
plt.xlabel('Time (seconds)')
plt.ylabel('Emotion Category')
plt.title('\'APT\' Emotion Profile Over Time (75%)')
plt.yticks([0, 1, 2, 3], ['Happy', 'Angry', 'Sad', 'Calm'])
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

emotions = ['25%', '50%', '75%', '100%']
counts = [26, 19, 52, 92]  # Example values, adjust as needed


# Create the horizontal bar graph
plt.figure(figsize=(8, 3))
plt.barh(emotions, counts,  height=.8)

# Labels and title
plt.xlabel("Accuracy (%)")
plt.ylabel("Configuration")
plt.title("Accuracy across the configurations")

plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Data setup
emotions = ['Happy', 'Angry', 'Sad', 'Calm']
configurations = ['25%', '50%', '75%', '100%']

# Example data for each configuration (replace with your actual data)
counts = {
    '25%': [23, 17, 7, 8],
    '50%': [21, 17, 6, 11],
    '75%': [24, 15, 11, 5],
    '100%': [18, 16, 8, 13]
}

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
fig.suptitle('Emotion Distribution Across Configurations', fontsize=14, y=1.02)

# Flatten axes for easy iteration
axes = axes.flatten()

# Plot each configuration
for ax, (config, values) in zip(axes, counts.items()):
    # Keep original orange color with alpha=0.7
    bars = ax.barh(emotions, values, height=0.6, alpha=0.7)
    ax.set_title(config, pad=10)
    ax.set_xlabel('Frequency')
    ax.set_xlim(0, max(max(v) for v in counts.values()) + 2)

    # Remove right/top borders for cleaner look
    ax.spines[['right', 'top']].set_visible(False)

    # Add value labels inside bars (left-aligned)
    for bar in bars:
        width = bar.get_width()
        ax.text(width - 0.5, bar.get_y() + bar.get_height()/2,
                f'{width}',
                ha='right', va='center',
                color='black', fontsize=10)

    # Add emotion labels on the right side
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(np.arange(len(emotions)))
    ax2.set_yticklabels(emotions, fontsize=10)
    ax2.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
    ax2.tick_params(left=False, right=False, labelleft=False, labelright=True)

# Adjust spacing
plt.tight_layout()
plt.subplots_adjust(hspace=0.3, wspace=0.15)
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Simulated data for four configurations (replace with your actual data)
emotion_profiles = {
    '25%':  emotion_changes_25,
    '50%': emotion_changes_50,
    '75%': emotion_changes_75,
    '100%': emotion_changes_100
}

seconds = [x * 3 for x in range(1, 56)]  # 3-second intervals
emotion_labels = ['Happy', 'Angry', 'Sad', 'Calm']

line_styles = {
    '25%': {'color': '#4CAF50', 'linestyle': 'dashdot', 'marker': 'o', 'linewidth': 2},
    '50%': {'color': '#FF5722', 'linestyle': 'dotted', 'marker': 's', 'linewidth': 2},
    '75%': {'color': '#2196F3', 'linestyle': 'dashed', 'marker': '^', 'linewidth': 2},
    '100%': {'color': '#9C27B0', 'linestyle': 'solid', 'marker': 'D', 'linewidth': 2}
}

# Set up the plot
plt.figure(figsize=(14, 6))

# Color palette for each configuration
colors = ['#4CAF50', '#FF5722', '#2196F3', '#9C27B0']  # Green, Orange, Blue, Purple

# Plot each configuration
for config, profile in emotion_profiles.items():
    style = line_styles[config]
    plt.plot(
        seconds,
        profile,
        label=f'{config} Configuration',
        color=style['color'],
        linestyle=style['linestyle'],
        marker=style['marker'],
        markersize=5,
        linewidth=style['linewidth'],
        alpha=0.8
    )

# Customize the plot
plt.xlabel('Time (seconds)', fontsize=12)
plt.ylabel('Emotion Category', fontsize=12)
plt.title('Emotion Profiles Across Configurations Over Time', fontsize=14, pad=20)
plt.yticks([0, 1, 2, 3], emotion_labels)

# Add legend inside upper right corner with semi-transparent background
legend = plt.legend(loc='upper right', framealpha=0.7)
legend.get_frame().set_facecolor('#FFFFFF')  # White background
legend.get_frame().set_edgecolor('#CCCCCC')  # Light gray border

plt.grid(True, linestyle='--', alpha=0.6)

# Adjust layout to prevent legend cutoff
plt.tight_layout()
plt.show()