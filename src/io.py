import librosa as lib
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import yaml
from pathlib import Path

# NOTE: not used by any notebook or module (checked 2026-11-16)
def load_dataset_csv(directory: str, annotation_file: str, subdirectory: str):
    """Load a dataset whose annotations are in a .csv file.

    Returns:
        (annotations, files, sounds, srl): the annotations DataFrame, the sorted .wav filenames,
        the audio signals and their sample rates (one element per file, in the same order).
    """
    annotations = pd.read_csv(f"{directory}/{annotation_file}")

    files = sorted((i for i in os.listdir(f"{directory}/{subdirectory}") if i.endswith(".wav")))
    sounds = np.empty(len(files), dtype=object)
    srl = np.empty(len(files), dtype=object)
    for n, i in enumerate(files):
        # sr=None keeps each file's original sample rate
        f, sr = lib.load(f"{directory}/{subdirectory}/{i}", sr=None)
        sounds[n] = f
        srl[n] = sr
    return annotations, files, sounds, srl

def load_dataset_xlsx(directory: str, annotation_file: str, subdirectory: str):
    """Load a dataset whose annotations are in an .xlsx file.

    Returns:
        (annotations, files, sounds, srl): the annotations DataFrame, the sorted .wav filenames,
        the audio signals and their sample rates (one element per file, in the same order).
    """
    annotations = pd.read_excel(f"{directory}/{annotation_file}")

    files = sorted((i for i in os.listdir(f"{directory}/{subdirectory}") if i.endswith(".wav")))
    sounds = np.empty(len(files), dtype=object)
    srl = np.empty(len(files), dtype=object)
    for n, i in enumerate(files):
        # sr=None keeps each file's original sample rate
        f, sr = lib.load(f"{directory}/{subdirectory}/{i}", sr=None)
        sounds[n] = f
        srl[n] = sr
    return annotations, files, sounds, srl

# NOTE: not used by any notebook or module (checked 2026-11-16)
def spectrogram_array(lungSounds: np.ndarray, lungFiles: list[str], lungSR: np.ndarray):
    """Plot one spectrogram (dB, linear frequency axis) per audio file.

    Every figure is drawn, but only the last one is returned.
    """
    config = open_config()
    for n, f in enumerate(lungSounds):
        fig = plt.figure(figsize=(6,4))
        # Convert amplitude to dB so quiet components remain visible
        spec = lib.amplitude_to_db(np.abs(lib.stft(f, n_fft=config['n-fft'])), ref=np.max)
        # hop_length must match the STFT, which uses librosa's default (n_fft // 4)
        img = lib.display.specshow(spec, sr=lungSR[n], hop_length=config['n-fft'] // 4,
                                x_axis="time", y_axis="linear")
        plt.title(f"{lungFiles[n]}, sr={lungSR[n]}")
    return fig

def spectrogram(file: np.ndarray, name: str, sample_rate: int | float):
    """Plot and return the spectrogram (dB, linear frequency axis) of a single signal."""
    config = open_config()
    fig = plt.figure(figsize=(6,4))
    # Convert amplitude to dB so quiet components remain visible
    spec = lib.amplitude_to_db(np.abs(lib.stft(file, n_fft=config['n-fft'])), ref=np.max)
    # hop_length must match the STFT, which uses librosa's default (n_fft // 4)
    img = lib.display.specshow(spec, sr=sample_rate, hop_length=config['n-fft'] // 4,
                            x_axis="time", y_axis="linear")
    plt.title(f"{name}, sr={sample_rate}")
    return fig

def open_config():
    """Load config/config.yaml and return it as a dict."""
    # Path built from this file's location, so it works from any working directory
    CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    return config
