import librosa
import numpy as np
import pandas as pd

df = pd.read_csv('../audio_features_25_soxrHQ_44_all_stacked_189.csv')
# Feature extraction functions
def extract_features(data, sample_rate):
    result = np.array([])

    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40).T, axis=0)
    result = np.hstack((result, mfcc))
    # print("mfcc: ", len(mfcc))

    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate, hop_length=20, n_mels=128).T, axis=0)
    result = np.hstack((result, mel))
    # print("mel: ", len(mel))

    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))
    # print("chrome: ", len(chroma_stft))

    rolloff = librosa.feature.spectral_rolloff(y=data, sr=sample_rate)[0]
    result = np.hstack((result, rolloff))
    # print("rolloff: ", len(rolloff))

    spec_bw = librosa.feature.spectral_bandwidth(y=data, sr=sample_rate)[0]
    result = np.hstack((result,spec_bw))
    # print("spec_bw: ", len(spec_bw)

    contrast = librosa.feature.spectral_contrast(y=data)[0]
    result = np.hstack((result, contrast))
    # print("contrast: ", len(contrast))

    flatness = librosa.feature.spectral_flatness(y=data)[0]
    result = np.hstack((result, flatness))
    # print("flatness: ", len(flatness))

    cent = librosa.feature.spectral_centroid(y=data, sr=sample_rate)[0]
    result = np.hstack((result, cent))

    rms =  librosa.feature.rms(y=data, frame_length=100)[0]
    result = np.hstack((result, rms))
    # print("rms: ", len(rms))

    tempo, _ = librosa.beat.beat_track(y=data, sr=sample_rate)
    result = np.hstack((result, tempo))
    # print("tempo: ", len(tempo))

    zcr = librosa.feature.zero_crossing_rate(y=data)[0]
    result = np.hstack((result, zcr))
    # print("zcr: ", len(zcr))

    return result

def feature_extractor(path):
    data, sample_rate = librosa.load(path, duration=25 ,sr=22050*2 ,offset=3)
    res1 = extract_features(data, sample_rate)
    result = np.array(res1)
    return result

X, y = [], []
for index, (data, emotion) in enumerate(zip(df["path"], df["label"])):
    feature = feature_extractor(data)
    X.append(feature)
    y.append(emotion)
    if index % 20 == 0:
        print("#", end="")

X = np.array(X)
dataframe = pd.DataFrame(data=X, columns=[str(x) for x in range(1, X.shape[1] + 1)])
dataframe["label"] = y
dataframe.to_csv('feature_extraction_output.csv', index=False)