import librosa
import numpy as np
import pandas as pd

# Feature extraction functions
def extractMFCC(y, sr):
    return np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)

def extractRMS(y, sr):
    return np.mean(librosa.feature.rms(y=y).T, axis=0)

def extractZCR(y, sr):
    return np.mean(librosa.feature.zero_crossing_rate(y=y).T, axis=0)

def extractChromaSTFT(y, sr):
    return np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)

def extractSpectralCentroid(y, sr):
    return np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)

def extractSpectralBandwidth(y, sr):
    return np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr).T, axis=0)

def extractSpectralRolloff(y, sr):
    return np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr).T, axis=0)

def extractMelSpectrogram(y, sr):
    return np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128).T, axis=0)

# Combine features into a single array
def extract_features(data, sample_rate):
    mfcc = extractMFCC(data, sample_rate)
    rms = extractRMS(data, sample_rate)
    zcr = extractZCR(data, sample_rate)
    chroma_stft = extractChromaSTFT(data, sample_rate)
    spectral_centroid = extractSpectralCentroid(data, sample_rate)
    spectral_bandwidth = extractSpectralBandwidth(data, sample_rate)
    spectral_rolloff = extractSpectralRolloff(data, sample_rate)
    mel_spectrogram = extractMelSpectrogram(data, sample_rate)
    return np.hstack([mfcc, rms, zcr, chroma_stft, spectral_centroid, spectral_bandwidth, spectral_rolloff, mel_spectrogram])

# Extract features for segmented audio
X = []
for segment in segments:
    features = extract_features(segment, 22050 * 2)
    X.append(features)

X = np.array(X)
dataframe = pd.DataFrame(data=X, columns=[str(x) for x in range(1, X.shape[1] + 1)])
dataframe.to_csv('/content/drive/MyDrive/thesis/dataset/songofruth_segments.csv', index=False)