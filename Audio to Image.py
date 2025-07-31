import os
import tensorflow as tf
import librosa
import numpy as np
import matplotlib.pyplot as plt

def audio_to_spectrogram(audio_path, output_folder):
    y, sr = librosa.load(audio_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_DB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(6, 4))  # Adjust size as needed
    ax = plt.Axes(plt.gcf(), [0., 0., 1., 1.])  # Fill the figure completely
    plt.gcf().add_axes(ax)
    ax.set_axis_off()

    # Draw the spectrogram only, no axis or border
    librosa.display.specshow(S_DB, sr=sr, x_axis=None, y_axis=None, cmap='magma')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.axis('off')
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    output_path = os.path.join(output_folder, f"{base_name}_spectrogram.png")
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Spectrogram saved to {output_path}")

cats_output_folder = 'valid/cats'
dogs_output_folder = 'valid/dogs'

os.makedirs(cats_output_folder, exist_ok=True)
os.makedirs(dogs_output_folder, exist_ok=True)

def process_audio_folder(source_folder, output_folder):
    for file in os.listdir(source_folder):
        if file.endswith('.wav') or file.endswith('.mp3'):
            audio_to_spectrogram(os.path.join(source_folder, file), output_folder)

# Set your actual source folders containing the audio files
cats_folder = '/home/devraj-bavan/Documents/Projects/Dog&Cat/Dogs&Cats Dataset-20250730T112123Z-1-002/Dogs&Cats Dataset/valid/cat'
dogs_folder = '/home/devraj-bavan/Documents/Projects/Dog&Cat/Dogs&Cats Dataset-20250730T112123Z-1-002/Dogs&Cats Dataset/valid/dog'

process_audio_folder(cats_folder, cats_output_folder)
process_audio_folder(dogs_folder, dogs_output_folder)

