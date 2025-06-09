import matplotlib.pyplot as plt
import numpy as np

# Accuracy comparison across models
models = ["SVM", "Random Forest", "Decision Tree", "Naive Bayes"]
accuracies = [svm_acc, rf_acc, dt_acc, nb_acc]

plt.figure(figsize=(8, 5))
plt.barh(models, accuracies, color=["blue", "green", "red", "orange"], height=0.6)
plt.xlabel("Accuracy (%)", fontsize=12)
plt.ylabel("Models", fontsize=12)
plt.title("Model Accuracy Comparison", fontsize=14)
for index, value in enumerate(accuracies):
    plt.text(value + 0.01, index, f"{value:.2f}", va='center', fontsize=10)
plt.grid(axis="x", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# Emotion distribution across configurations
configurations = ['25%', '50%', '75%', '100%']
emotion_counts = {
    '25%': [23, 17, 7, 8],
    '50%': [21, 17, 6, 11],
    '75%': [24, 15, 11, 5],
    '100%': [18, 16, 8, 13]
}
emotions = ['Happy', 'Angry', 'Sad', 'Calm']

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
axes = axes.flatten()
for ax, (config, values) in zip(axes, emotion_counts.items()):
    ax.barh(emotions, values, height=0.6, alpha=0.7, color=['#4CAF50', '#FF5722', '#2196F3', '#9C27B0'])
    ax.set_title(f"{config} Configuration", fontsize=12)
    ax.set_xlabel('Frequency', fontsize=10)
    ax.set_xlim(0, max(max(v) for v in emotion_counts.values()) + 2)
plt.suptitle('Emotion Distribution Across Configurations', fontsize=14, y=1.02)
plt.tight_layout()
plt.subplots_adjust(hspace=0.3, wspace=0.15)
plt.show()

# Emotion profiles over time
emotion_profiles = {
    '25%': [0, 1, 2, 3],
    '50%': [1, 2, 3, 0],
    '75%': [2, 3, 0, 1],
    '100%': [3, 0, 1, 2]
}
seconds = [x * 3 for x in range(1, 56)]  # 3-second intervals
emotion_labels = ['Happy', 'Angry', 'Sad', 'Calm']

line_styles = {
    '25%': {'color': '#4CAF50', 'linestyle': 'dashdot', 'marker': 'o', 'linewidth': 2},
    '50%': {'color': '#FF5722', 'linestyle': 'dotted', 'marker': 's', 'linewidth': 2},
    '75%': {'color': '#2196F3', 'linestyle': 'dashed', 'marker': '^', 'linewidth': 2},
    '100%': {'color': '#9C27B0', 'linestyle': 'solid', 'marker': 'D', 'linewidth': 2}
}

plt.figure(figsize=(14, 6))
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
plt.xlabel('Time (seconds)', fontsize=12)
plt.ylabel('Emotion Category', fontsize=12)
plt.title('Emotion Profiles Across Configurations Over Time', fontsize=14, pad=20)
plt.yticks([0, 1, 2, 3], emotion_labels)
legend = plt.legend(loc='upper right', framealpha=0.7)
legend.get_frame().set_facecolor('#FFFFFF')  # White background
legend.get_frame().set_edgecolor('#CCCCCC')  # Light gray border
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Horizontal bar graph for configuration accuracy
emotions = ['25%', '50%', '75%', '100%']
counts = [26, 19, 52, 92]  # Example values, adjust as needed

plt.figure(figsize=(8, 3))
plt.barh(emotions, counts, height=0.8, color=['#4CAF50', '#FF5722', '#2196F3', '#9C27B0'])
plt.xlabel("Accuracy (%)", fontsize=12)
plt.ylabel("Configuration", fontsize=12)
plt.title("Accuracy Across Configurations", fontsize=14)
plt.tight_layout()
plt.show()