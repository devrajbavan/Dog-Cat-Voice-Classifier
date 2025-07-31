import os
import librosa
import matplotlib.pyplot as plt
import librosa.display
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# Paths (update as needed)
MODEL_PATH = 'meow_vs_Bark.h5'
AUDIO_DIR = 'test'            # Folder containing all the audio files
TEMP_SPEC_DIR = 'temp_specs'        # Temporary spectrogram images
LABELS_PATH = 'TrueLabels.txt'      # Your labels file

# Read true labels file (space-separated, first col = label [0/1], second col = filename)
def load_labels(labels_path):
    true_labels = []
    file_names = []
    with open(labels_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                label, fname = parts
                true_labels.append(int(label))
                file_names.append(fname)
    return true_labels, file_names

def audio_to_spectrogram(audio_path, output_path):
    y, sr = librosa.load(audio_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_DB = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(3, 3))
    ax = plt.Axes(plt.gcf(), [0., 0., 1., 1.])
    plt.gcf().add_axes(ax)
    ax.set_axis_off()
    librosa.display.specshow(S_DB, sr=sr, x_axis=None, y_axis=None, cmap='magma')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def predict_spectrogram(image_path, model, image_size=(300, 300)):
    img = image.load_img(image_path, target_size=image_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    return int(pred[0][0] > 0.5), float(pred[0][0])

def main():
    os.makedirs(TEMP_SPEC_DIR, exist_ok=True)
    # Read true labels and filenames
    y_true, file_names = load_labels(LABELS_PATH)
    model = tf.keras.models.load_model(MODEL_PATH)
    y_pred = []

    for idx, fname in enumerate(file_names):
        audio_path = os.path.join(AUDIO_DIR, fname)
        if not os.path.exists(audio_path):
            print(f"Audio file not found: {audio_path}. Skipping.")
            y_pred.append(0)  # Default to cat if missing, or skip altogether
            continue

        spec_path = os.path.join(TEMP_SPEC_DIR, f"{os.path.splitext(fname)[0]}.png")
        audio_to_spectrogram(audio_path, spec_path)
        pred_label, confidence = predict_spectrogram(spec_path, model)
        label_word = 'Dog' if pred_label else 'Cat'
        print(f'File: {fname} → Prediction: {label_word} (Confidence: {confidence:.4f})')
        y_pred.append(pred_label)
        os.remove(spec_path)

    if os.path.exists(TEMP_SPEC_DIR) and not os.listdir(TEMP_SPEC_DIR):
        os.rmdir(TEMP_SPEC_DIR)

    # Confusion matrix and report
    print("\nConfusion Matrix (rows: True [Cat, Dog], cols: Predicted [Cat, Dog]):")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Cat', 'Dog']))

if __name__ == "__main__":
    main()

