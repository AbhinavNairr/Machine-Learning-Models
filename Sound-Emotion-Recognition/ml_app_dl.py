from pathlib import Path

import librosa
import numpy as np
import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "sound_emotion_model.keras"
IMAGE_DIR = PROJECT_DIR / "images"
EMOTION_LABELS = np.array(
    ["angry", "calm", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def noise(data):
    noise_amp = 0.035 * np.random.uniform() * np.amax(data)
    return data + noise_amp * np.random.normal(size=data.shape[0])


def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate=rate)


def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(
        y=data, sr=sampling_rate, n_steps=int(pitch_factor)
    )


def extract_features(data, sample_rate):
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)

    stft = np.abs(librosa.stft(data))
    chroma = np.mean(
        librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0
    )
    contrast = np.mean(
        librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0
    )
    return np.hstack((mfcc, rms, mel, zcr, chroma, contrast))


def get_features(path):
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    original = extract_features(data, sample_rate)
    noisy = extract_features(noise(data), sample_rate)
    stretched = stretch(data)
    stretched_pitch = extract_features(
        pitch(stretched, sample_rate), sample_rate
    )
    return np.vstack((original, noisy, stretched_pitch))


model = load_model()
scaler = StandardScaler()

st.title("Sound Emotion Recognition App")
st.subheader("Deep Learning model")
st.image(str(IMAGE_DIR / "img_emotion.png"))

audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if audio_file is not None:
    st.audio(audio_file, format="audio/wav")
    audio_path = PROJECT_DIR / "uploaded_audio.wav"
    audio_path.write_bytes(audio_file.getbuffer())

    features = scaler.fit_transform(get_features(str(audio_path)))
    probabilities = model.predict(features, verbose=0).mean(axis=0)
    prediction_index = int(np.argmax(probabilities))

    st.write(f"Prediction: {EMOTION_LABELS[prediction_index]}")
    st.write(f"Confidence: {probabilities[prediction_index]:.2%}")

st.image(str(IMAGE_DIR / "Speech_Emotion_Recognition.webp"))
