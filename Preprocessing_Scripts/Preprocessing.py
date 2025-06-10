import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Paths
path_audio = '/audio/'
data_path = '/audio_dataset.csv'

# Load dataset
data = pd.read_csv(data_path)
df = pd.DataFrame(data)

# Create audio file paths
def createPath(code):
    return path_audio + str(code) + '.mp3'

df['path'] = df['Code'].apply(createPath)
df.drop('Code', axis=1, inplace=True)

# Encode labels
lb = LabelEncoder()
df['label'] = lb.fit_transform(df['Label'])
df.drop('Label', axis=1, inplace=True)

# Remove invalid audio files
def isFileExists(path):
    try:
        y, sr = librosa.load(path, res_type='soxr_lq', duration=1, sr=22050, offset=5)
    except:
        df.drop(df[df['path'] == path].index, inplace=True)

df['path'].apply(isFileExists)

# Visualize data distribution
classes_dict = {0: 'Happy', 1: 'Angry', 2: 'Sad', 3: 'Calm'}
p = df['label'].value_counts()
classes = ["Happy", "Angry", "Sad", "Calm"]
values = list(p)

plt.figure(figsize=(10, 5))
plt.bar(classes, values, color='green', width=0.4)
plt.xlabel("Emotion")
plt.ylabel("Number of Songs")
plt.title("Number of Songs by Emotion")
plt.show()

# Segment audio into smaller chunks
def segment_audio(audio, sr, seconds, overlap):
    if seconds <= 0 or overlap < 0 or seconds <= overlap:
        raise ValueError("Invalid seconds or overlap parameters.")
    seconds = int(seconds * sr)
    overlap = int(overlap * sr)
    step = seconds - overlap
    segments = [audio[i:i + seconds] for i in range(0, len(audio) - seconds + 1, step)]
    return segments

def getSegments():
    path_audio_full = '/audio/songofruth.mp3'
    y, sr = librosa.load(path_audio_full, sr=22050 * 2)
    y, _ = librosa.effects.trim(y)
    return segment_audio(y, sr, seconds=5, overlap=2)

segments = getSegments()