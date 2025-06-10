happy = df[df['label'] == 0].iloc[0]['path']
angry = df[df['label'] == 1].iloc[0]['path']
sad = df[df['label'] == 2].iloc[0]['path']
calm = df[df['label'] == 3].iloc[0]['path']

y1, sr = librosa.load(happy, res_type='soxr_lq',duration=25 ,sr=22050 * 2 ,offset=3)

y2, sr = librosa.load(angry, res_type='soxr_lq',duration=25 ,sr=22050 * 2 ,offset=3)

y3, sr = librosa.load(sad, res_type='soxr_lq',duration=25 ,sr=22050 * 2 ,offset=3)

y4, sr = librosa.load(calm, res_type='soxr_lq',duration=25 ,sr=22050 * 2 ,offset=3)

from scipy.stats import skew

# Generate left-skewed data (beta distribution with alpha > beta)
left_skewed_data = np.exp(-data)  # Exponential flip

# Method 2: Reflection (quick left skew)
left_skewed_data = np.max(data) - data + np.min(data)

# Plot
plt.hist(left_skewed_data, bins=30, density=True, alpha=0.6, color='salmon')
plt.title("Artificially Left-Skewed Data (Exponential Transform)")
plt.show()
plt.show()

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

import math

contrast1 = librosa.feature.spectral_contrast(y=y1)[0]
contrast2 = librosa.feature.spectral_contrast(y=y2)[0]
contrast3 = librosa.feature.spectral_contrast(y=y3)[0]
contrast4 = librosa.feature.spectral_contrast(y=y4)[0]

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle('Spectral Contrast Values Sorted by Frequency per Emotion', fontsize=16)

emotions_contrast = {
    'happy': contrast1,
    'angry': contrast2,
    'sad': contrast3,
    'calm': contrast4
}
# Define colors
emotion_colors = {
    'happy': 'limegreen',
    'sad': 'dodgerblue',
    'angry': 'tomato',
    'calm': 'gold'
}

for (emotion, rms_data), ax in zip(emotions_contrast.items(), axes.flat):
    # 1. Compute histogram
    counts, bin_edges = np.histogram(rms_data, bins=50)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    # 2. Sort by frequency (descending)
    sorted_indices = np.argsort(counts)[::-1]
    sorted_counts = counts[sorted_indices]
    sorted_bins = bin_centers[sorted_indices]

    # 3. Plot sorted bars
    bars = ax.bar(
        range(len(sorted_counts)),
        sorted_counts,
        width=1,
        color=emotion_colors[emotion],
        edgecolor='black',
        alpha=0.7
    )

    # # 4. Highlight top 25% most frequent bins
    # top_25_cutoff = math.floor(len(sorted_counts) * 0.25)
    # for i in range(top_25_cutoff):
    #     bars[i].set_color('red')
    #     bars[i].set_alpha(0.9)

    # 5. Customize subplot
    ax.set_title(f'{emotion.capitalize()}', fontweight='bold')
    ax.set_xlabel('Spectral Contrast Bins')
    ax.set_ylabel('Frequency')
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    # 6. Clean x-axis labels (show every 5th bin)
    ax.set_xticks(range(0, len(sorted_counts), 5))
    ax.set_xticklabels([f"{val:.2f}" for val in sorted_bins[::5]], rotation=45)

plt.tight_layout()
plt.show()

flatnes1 = librosa.feature.spectral_flatness(y=y1)[0]
flatnes2 = librosa.feature.spectral_flatness(y=y2)[0]
flatnes3 = librosa.feature.spectral_flatness(y=y3)[0]
flatnes4 = librosa.feature.spectral_flatness(y=y4)[0]

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle('Spectral Flatness Values Sorted by Frequency per Emotion', fontsize=16)

emotions_flatness ={
    'happy': flatnes1,
    'angry': flatnes2,
    'sad': flatnes3,
    'calm': flatnes4
}

# Combine all RMS values
all_rms = np.concatenate(list(emotions_flatness.values())).reshape(-1, 1)

# Fit scaler globally
scaler = MinMaxScaler(feature_range=(0, 100)).fit(all_rms)

# Transform each emotion
emotions_flatness_scaled = {
    emotion: scaler.transform(values.reshape(-1, 1)).flatten()
    for emotion, values in emotions_flatness.items()
}

# Define colors
emotion_colors = {
    'happy': 'limegreen',
    'sad': 'dodgerblue',
    'angry': 'tomato',
    'calm': 'gold'
}

for (emotion, rms_data), ax in zip(emotions_flatness_scaled.items(), axes.flat):
    # 1. Compute histogram
    counts, bin_edges = np.histogram(rms_data, bins=50)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    # 2. Sort by frequency (descending)
    sorted_indices = np.argsort(counts)[::-1]
    sorted_counts = counts[sorted_indices]
    sorted_bins = bin_centers[sorted_indices]

    # 3. Plot sorted bars
    bars = ax.bar(
        range(len(sorted_counts)),
        sorted_counts,
        width=1,
        color=emotion_colors[emotion],
        edgecolor='black',
        alpha=0.7
    )

    # # 4. Highlight top 25% most frequent bins
    # top_25_cutoff = math.floor(len(sorted_counts) * 0.25)
    # for i in range(top_25_cutoff):
    #     bars[i].set_color('red')
    #     bars[i].set_alpha(0.9)

    # 5. Customize subplot
    ax.set_title(f'{emotion.capitalize()}', fontweight='bold')
    ax.set_xlabel('Spectral Flatness Bins')
    ax.set_ylabel('Frequency')
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    # 6. Clean x-axis labels (show every 5th bin)
    ax.set_xticks(range(0, len(sorted_counts), 5))
    ax.set_xticklabels([f"{val:.2f}" for val in sorted_bins[::5]], rotation=45)

plt.tight_layout()
plt.show()



zcr1 = librosa.feature.zero_crossing_rate(y=y1)[0]
zcr2 = librosa.feature.zero_crossing_rate(y=y2)[0]
zcr3 = librosa.feature.zero_crossing_rate(y=y3)[0]
zcr4 = librosa.feature.zero_crossing_rate(y=y4)[0]

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle('ZCR Values Sorted by Frequency per Emotion', fontsize=16)

# Define colors
emotion_colors = {
    'happy': 'limegreen',
    'sad': 'dodgerblue',
    'angry': 'tomato',
    'calm': 'gold'
}

for (emotion, rms_data), ax in zip(emotions_rms.items(), axes.flat):
    # 1. Compute histogram
    counts, bin_edges = np.histogram(rms_data, bins=50)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    # 2. Sort by frequency (descending)
    sorted_indices = np.argsort(counts)[::-1]
    sorted_counts = counts[sorted_indices]
    sorted_bins = bin_centers[sorted_indices]

    # 3. Plot sorted bars
    bars = ax.bar(
        range(len(sorted_counts)),
        sorted_counts,
        width=1,
        color=emotion_colors[emotion],
        edgecolor='black',
        alpha=0.7
    )

    # # 4. Highlight top 25% most frequent bins
    # top_25_cutoff = math.floor(len(sorted_counts) * 0.25)
    # for i in range(top_25_cutoff):
    #     bars[i].set_color('red')
    #     bars[i].set_alpha(0.9)

    # 5. Customize subplot
    ax.set_title(f'{emotion.capitalize()}', fontweight='bold')
    ax.set_xlabel('ZCR Bins')
    ax.set_ylabel('Frequency')
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    # 6. Clean x-axis labels (show every 5th bin)
    ax.set_xticks(range(0, len(sorted_counts), 5))
    ax.set_xticklabels([f"{val:.2f}" for val in sorted_bins[::5]], rotation=45)

plt.tight_layout()
plt.show()

mfcc1 = librosa.feature.mfcc(y=y1, sr=sr, n_mfcc=40)[0]
mfcc2 = librosa.feature.mfcc(y=y2, sr=sr, n_mfcc=40)[0]
mfcc3 = librosa.feature.mfcc(y=y3, sr=sr, n_mfcc=40)[0]
mfcc4 = librosa.feature.mfcc(y=y4, sr=sr, n_mfcc=40)[0]

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle('MFCC Values Sorted by Frequency per Emotion', fontsize=16)

emotions_mfccs ={
    'happy': mfcc1,
    'angry': mfcc2,
    'sad': mfcc3,
    'calm': mfcc4
}

# Define colors
emotion_colors = {
    'happy': 'limegreen',
    'sad': 'dodgerblue',
    'angry': 'tomato',
    'calm': 'gold'
}

for (emotion, rms_data), ax in zip(emotions_mfccs.items(), axes.flat):
    # 1. Compute histogram
    counts, bin_edges = np.histogram(rms_data, bins=50)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    # 2. Sort by frequency (descending)
    sorted_indices = np.argsort(counts)[::-1]
    sorted_counts = counts[sorted_indices]
    sorted_bins = bin_centers[sorted_indices]

    # 3. Plot sorted bars
    bars = ax.bar(
        range(len(sorted_counts)),
        sorted_counts,
        width=1,
        color=emotion_colors[emotion],
        edgecolor='black',
        alpha=0.7
    )

    # # 4. Highlight top 25% most frequent bins
    # top_25_cutoff = math.floor(len(sorted_counts) * 0.25)
    # for i in range(top_25_cutoff):
    #     bars[i].set_color('red')
    #     bars[i].set_alpha(0.9)

    # 5. Customize subplot
    ax.set_title(f'{emotion.capitalize()}', fontweight='bold')
    ax.set_xlabel('MFCC Bins')
    ax.set_ylabel('Frequency')
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    # 6. Clean x-axis labels (show every 5th bin)
    ax.set_xticks(range(0, len(sorted_counts), 5))
    ax.set_xticklabels([f"{val:.2f}" for val in sorted_bins[::5]], rotation=45)

plt.tight_layout()
plt.show()

stft = np.abs(librosa.stft(y1))
chroma1 = librosa.feature.chroma_stft(S=stft, sr=sr)
stft = np.abs(librosa.stft(y2))
chroma2 = librosa.feature.chroma_stft(S=stft, sr=sr)
stft = np.abs(librosa.stft(y3))
chroma3 = librosa.feature.chroma_stft(S=stft, sr=sr)
stft = np.abs(librosa.stft(y4))
chroma4 = librosa.feature.chroma_stft(S=stft, sr=sr)

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle('Chroma Values Sorted by Frequency per Emotion', fontsize=16)

emotions_chroma ={
    'happy': chroma1,
    'angry': chroma2,
    'sad': chroma3,
    'calm': chroma4
}

# Define colors
emotion_colors = {
    'happy': 'limegreen',
    'sad': 'dodgerblue',
    'angry': 'tomato',
    'calm': 'gold'
}

for (emotion, rms_data), ax in zip(emotions_chroma.items(), axes.flat):
    # 1. Compute histogram
    counts, bin_edges = np.histogram(rms_data, bins=50)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    # 2. Sort by frequency (descending)
    sorted_indices = np.argsort(counts)[::-1]
    sorted_counts = counts[sorted_indices]
    sorted_bins = bin_centers[sorted_indices]

    # 3. Plot sorted bars
    bars = ax.bar(
        range(len(sorted_counts)),
        sorted_counts,
        width=1,
        color=emotion_colors[emotion],
        edgecolor='black',
        alpha=0.7
    )

    # # 4. Highlight top 25% most frequent bins
    # top_25_cutoff = math.floor(len(sorted_counts) * 0.25)
    # for i in range(top_25_cutoff):
    #     bars[i].set_color('red')
    #     bars[i].set_alpha(0.9)

    # 5. Customize subplot
    ax.set_title(f'{emotion.capitalize()}', fontweight='bold')
    ax.set_xlabel('Spectral Flatness Bins')
    ax.set_ylabel('Frequency')
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    # 6. Clean x-axis labels (show every 5th bin)
    ax.set_xticks(range(0, len(sorted_counts), 5))
    ax.set_xticklabels([f"{val:.2f}" for val in sorted_bins[::5]], rotation=45)

plt.tight_layout()
plt.show()

import math

bandwidth1 = librosa.feature.spectral_bandwidth(y=y1)[0]
bandwidth2 = librosa.feature.spectral_bandwidth(y=y2)[0]
bandwidth3 = librosa.feature.spectral_bandwidth(y=y3)[0]
bandwidth4 = librosa.feature.spectral_bandwidth(y=y4)[0]

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle('Spectral bandwidth Values Sorted by Frequency per Emotion', fontsize=16)

emotions_bandwidth = {
    'happy': bandwidth1,
    'angry': bandwidth2,
    'sad': bandwidth3,
    'calm': bandwidth4
}
# Define colors
emotion_colors = {
    'happy': 'limegreen',
    'sad': 'dodgerblue',
    'angry': 'tomato',
    'calm': 'gold'
}

for (emotion, rms_data), ax in zip(emotions_bandwidth.items(), axes.flat):
    # 1. Compute histogram
    counts, bin_edges = np.histogram(rms_data, bins=50)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    # 2. Sort by frequency (descending)
    sorted_indices = np.argsort(counts)[::-1]
    sorted_counts = counts[sorted_indices]
    sorted_bins = bin_centers[sorted_indices]

    # 3. Plot sorted bars
    bars = ax.bar(
        range(len(sorted_counts)),
        sorted_counts,
        width=1,
        color=emotion_colors[emotion],
        edgecolor='black',
        alpha=0.7
    )

    # # 4. Highlight top 25% most frequent bins
    # top_25_cutoff = math.floor(len(sorted_counts) * 0.25)
    # for i in range(top_25_cutoff):
    #     bars[i].set_color('red')
    #     bars[i].set_alpha(0.9)

    # 5. Customize subplot
    ax.set_title(f'{emotion.capitalize()}', fontweight='bold')
    ax.set_xlabel('Spectral Bandwidth Bins')
    ax.set_ylabel('Frequency')
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    # 6. Clean x-axis labels (show every 5th bin)
    ax.set_xticks(range(0, len(sorted_counts), 5))
    ax.set_xticklabels([f"{val:.2f}" for val in sorted_bins[::5]], rotation=45)

plt.tight_layout()
plt.show()

import math

# Assuming your data is structured like this (replace with your actual data):
# emotions_rms = {
#     'happy': [...],
#     'sad': [...],
#     'angry': [...],
#     'neutral': [...]
# }

rms1 = librosa.feature.rms(y=y1)[0]
rms2 = librosa.feature.rms(y=y2)[0]
rms3 = librosa.feature.rms(y=y3)[0]
rms4 = librosa.feature.rms(y=y4)[0]

emotions_rms = { # Corrected variable name
    'happy': rms1,
    'angry': rms2,
    'sad': rms3,
    'calm': rms4
}
# Create figure
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle('RMS Values Sorted by Frequency per Emotion', fontsize=16)

# Define colors
emotion_colors = {
    'happy': 'limegreen',
    'sad': 'dodgerblue',
    'angry': 'tomato',
    'calm': 'gold'
}

for (emotion, rms_data), ax in zip(emotions_rms.items(), axes.flat):
    # 1. Compute histogram
    counts, bin_edges = np.histogram(rms_data, bins=50)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    # 2. Sort by frequency (descending)
    sorted_indices = np.argsort(counts)[::-1]
    sorted_counts = counts[sorted_indices]
    sorted_bins = bin_centers[sorted_indices]

    # 3. Plot sorted bars
    bars = ax.bar(
        range(len(sorted_counts)),
        sorted_counts,
        width=1,
        color=emotion_colors[emotion],
        edgecolor='black',
        alpha=0.7
    )

    # # 4. Highlight top 25% most frequent bins
    # top_25_cutoff = math.floor(len(sorted_counts) * 0.25)
    # for i in range(top_25_cutoff):
    #     bars[i].set_color('red')
    #     bars[i].set_alpha(0.9)

    # 5. Customize subplot
    ax.set_title(f'{emotion.capitalize()}', fontweight='bold')
    ax.set_xlabel('RMS Bins (Sorted)')
    ax.set_ylabel('Frequency')
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    # 6. Clean x-axis labels (show every 5th bin)
    ax.set_xticks(range(0, len(sorted_counts), 5))
    ax.set_xticklabels([f"{val:.2f}" for val in sorted_bins[::5]], rotation=45)

plt.tight_layout()
plt.show()

import math

centroid1 = librosa.feature.spectral_centroid(y=y1)[0]
centroid2 = librosa.feature.spectral_centroid(y=y2)[0]
centroid3 = librosa.feature.spectral_centroid(y=y3)[0]
centroid4 = librosa.feature.spectral_centroid(y=y4)[0]

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle('Spectral centroid Values Sorted by Frequency per Emotion', fontsize=16)

emotions_centroid = {
    'happy': centroid1,
    'angry': centroid2,
    'sad': centroid3,
    'calm': centroid4
}
# Define colors
emotion_colors = {
    'happy': 'limegreen',
    'sad': 'dodgerblue',
    'angry': 'tomato',
    'calm': 'gold'
}

for (emotion, rms_data), ax in zip(emotions_centroid.items(), axes.flat):
    # 1. Compute histogram
    counts, bin_edges = np.histogram(rms_data, bins=50)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    # 2. Sort by frequency (descending)
    sorted_indices = np.argsort(counts)[::-1]
    sorted_counts = counts[sorted_indices]
    sorted_bins = bin_centers[sorted_indices]

    # 3. Plot sorted bars
    bars = ax.bar(
        range(len(sorted_counts)),
        sorted_counts,
        width=1,
        color=emotion_colors[emotion],
        edgecolor='black',
        alpha=0.7
    )

    # # 4. Highlight top 25% most frequent bins
    # top_25_cutoff = math.floor(len(sorted_counts) * 0.25)
    # for i in range(top_25_cutoff):
    #     bars[i].set_color('red')
    #     bars[i].set_alpha(0.9)

    # 5. Customize subplot
    ax.set_title(f'{emotion.capitalize()}', fontweight='bold')
    ax.set_xlabel('Spectral Centroid Bins')
    ax.set_ylabel('Frequency')
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    # 6. Clean x-axis labels (show every 5th bin)
    ax.set_xticks(range(0, len(sorted_counts), 5))
    ax.set_xticklabels([f"{val:.2f}" for val in sorted_bins[::5]], rotation=45)

plt.tight_layout()
plt.show()

import math

rolloff1 = librosa.feature.spectral_rolloff(y=y1)[0]
rolloff2 = librosa.feature.spectral_rolloff(y=y2)[0]
rolloff3 = librosa.feature.spectral_rolloff(y=y3)[0]
rolloff4 = librosa.feature.spectral_rolloff(y=y4)[0]

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle('Spectral rolloff Values Sorted by Frequency per Emotion', fontsize=16)

emotions_rolloff = {
    'happy': rolloff1,
    'angry': rolloff2,
    'sad': rolloff3,
    'calm': rolloff4
}
# Define colors
emotion_colors = {
    'happy': 'limegreen',
    'sad': 'dodgerblue',
    'angry': 'tomato',
    'calm': 'gold'
}

for (emotion, rms_data), ax in zip(emotions_rolloff.items(), axes.flat):
    # 1. Compute histogram
    counts, bin_edges = np.histogram(rms_data, bins=50)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    # 2. Sort by frequency (descending)
    sorted_indices = np.argsort(counts)[::-1]
    sorted_counts = counts[sorted_indices]
    sorted_bins = bin_centers[sorted_indices]

    # 3. Plot sorted bars
    bars = ax.bar(
        range(len(sorted_counts)),
        sorted_counts,
        width=1,
        color=emotion_colors[emotion],
        edgecolor='black',
        alpha=0.7
    )

    # # 4. Highlight top 25% most frequent bins
    # top_25_cutoff = math.floor(len(sorted_counts) * 0.25)
    # for i in range(top_25_cutoff):
    #     bars[i].set_color('red')
    #     bars[i].set_alpha(0.9)

    # 5. Customize subplot
    ax.set_title(f'{emotion.capitalize()}', fontweight='bold')
    ax.set_xlabel('Spectral Rolloff Bins')
    ax.set_ylabel('Frequency')
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    # 6. Clean x-axis labels (show every 5th bin)
    ax.set_xticks(range(0, len(sorted_counts), 5))
    ax.set_xticklabels([f"{val:.2f}" for val in sorted_bins[::5]], rotation=45)

plt.tight_layout()
plt.show()

import math

tempo1 = librosa.beat.beat_track(y=y1)[0]
tempo2 = librosa.beat.beat_track(y=y2)[0]
tempo3 = librosa.beat.beat_track(y=y3)[0]
tempo4 = librosa.beat.beat_track(y=y4)[0]

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle('Tempo Values Sorted by Frequency per Emotion', fontsize=16)

emotions_tempo = {
    'happy': tempo1,
    'angry': tempo2,
    'sad': tempo3,
    'calm': tempo4
}
# Define colors
emotion_colors = {
    'happy': 'limegreen',
    'sad': 'dodgerblue',
    'angry': 'tomato',
    'calm': 'gold'
}

for (emotion, rms_data), ax in zip(emotions_tempo.items(), axes.flat):
    # 1. Compute histogram
    counts, bin_edges = np.histogram(rms_data, bins=50)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    # 2. Sort by frequency (descending)
    sorted_indices = np.argsort(counts)[::-1]
    sorted_counts = counts[sorted_indices]
    sorted_bins = bin_centers[sorted_indices]

    # 3. Plot sorted bars
    bars = ax.bar(
        range(len(sorted_counts)),
        sorted_counts,
        width=1,
        color=emotion_colors[emotion],
        edgecolor='black',
        alpha=0.7
    )

    # # 4. Highlight top 25% most frequent bins
    # top_25_cutoff = math.floor(len(sorted_counts) * 0.25)
    # for i in range(top_25_cutoff):
    #     bars[i].set_color('red')
    #     bars[i].set_alpha(0.9)

    # 5. Customize subplot
    ax.set_title(f'{emotion.capitalize()}', fontweight='bold')
    ax.set_xlabel('Tempo Bins')
    ax.set_ylabel('Frequency')
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    # 6. Clean x-axis labels (show every 5th bin)
    ax.set_xticks(range(0, len(sorted_counts), 5))
    ax.set_xticklabels([f"{val:.2f}" for val in sorted_bins[::5]], rotation=45)

plt.tight_layout()
plt.show()

mel1 = librosa.feature.melspectrogram(y=y1, n_mels=128)[0]
mel2 = librosa.feature.melspectrogram(y=y2, n_mels=128)[0]
mel3 = librosa.feature.melspectrogram(y=y3, n_mels=128)[0]
mel4 = librosa.feature.melspectrogram(y=y4, n_mels=128)[0]
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle('Mel-scaled Spectrograms Values Sorted by Frequency per Emotion', fontsize=16)

emotions_mels ={
    'happy': mel1,
    'angry': mel2,
    'sad': mel3,
    'calm': mel4
}

# Define colors
emotion_colors = {
    'happy': 'limegreen',
    'sad': 'dodgerblue',
    'angry': 'tomato',
    'calm': 'gold'
}

for (emotion, rms_data), ax in zip(emotions_mels.items(), axes.flat):
    # 1. Compute histogram
    counts, bin_edges = np.histogram(rms_data, bins=50)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    # 2. Sort by frequency (descending)
    sorted_indices = np.argsort(counts)[::-1]
    sorted_counts = counts[sorted_indices]
    sorted_bins = bin_centers[sorted_indices]

    # 3. Plot sorted bars
    bars = ax.bar(
        range(len(sorted_counts)),
        sorted_counts,
        width=1,
        color=emotion_colors[emotion],
        edgecolor='black',
        alpha=0.7
    )

    # # 4. Highlight top 25% most frequent bins
    # top_25_cutoff = math.floor(len(sorted_counts) * 0.25)
    # for i in range(top_25_cutoff):
    #     bars[i].set_color('red')
    #     bars[i].set_alpha(0.9)

    # 5. Customize subplot
    ax.set_title(f'{emotion.capitalize()}', fontweight='bold')
    ax.set_xlabel('Mel-scaled Spectrograms Bins')
    ax.set_ylabel('Frequency')
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    # 6. Clean x-axis labels (show every 5th bin)
    ax.set_xticks(range(0, len(sorted_counts), 5))
    ax.set_xticklabels([f"{val:.2f}" for val in sorted_bins[::5]], rotation=45)

plt.tight_layout()
plt.show()

#this is to remove data that has no audio file
for index, (data, emotion) in enumerate(zip(df["path"], df["label"])):
  if index % 20 == 0:
      print("#", end="")
  try:
    y, sr = librosa.load(data, duration=1 ,sr=10 ,offset=3)
  except:
    df.drop(df[df['path'] == data].index, inplace=True)
    continue

#'Chroma', 'MFCC', 'Spectral Contrast', 'Spectral Flatness', 'ZCR'
import math