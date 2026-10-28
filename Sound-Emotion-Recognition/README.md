
# Deep Learning Sound Emotion Recognition

This project recognizes emotions in speech from the RAVDESS dataset using a
TensorFlow/Keras deep-learning model. The Streamlit application accepts WAV or
MP3 files, extracts 169 audio features, and predicts one of eight emotions:

`angry`, `calm`, `disgust`, `fear`, `happy`, `neutral`, `sad`, or `surprise`.

## Model

- Model artifact: [`sound_emotion_model.keras`](./sound_emotion_model.keras)
- Application: [`ml_app_dl.py`](./ml_app_dl.py)
- Feature extraction: MFCC, RMS, mel spectrogram, zero-crossing rate, chroma,
  and spectral contrast
- Inference: predictions are averaged across the original, noisy, and
  augmented audio variants
- Training: validation-based early stopping restores the best model weights

## Dataset

The model uses the
[RAVDESS emotional speech dataset]([https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio/data](https://www.kaggle.com/datasets/deepathangadurai/voice-emotion-regconition)).

## Installation

From this directory, install the dependencies listed in
[`requirements.txt`](./requirements.txt):

```bash
python -m pip install -r requirements.txt
```

For GPU inference or training, use an NVIDIA CUDA-enabled Python environment.
The project requirements include TensorFlow CUDA support:

```bash
python -m pip install "tensorflow[and-cuda]"
```

The optional RAPIDS packages enable GPU acceleration for supported pandas and
scikit-learn operations:

```python
%load_ext cudf.pandas
%load_ext cuml.accel
```

## Run the Streamlit app

Use the project environment that contains TensorFlow and the CUDA libraries:

```bash
streamlit run ml_app_dl.py
```

Upload a WAV or MP3 file in the browser. The app displays the audio,
predicted emotion, and prediction confidence.

## Output

![Streamlit interface](./images/Output.png)
