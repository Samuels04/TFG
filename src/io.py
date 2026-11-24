from typing import Any

import librosa as lib
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import tomllib
from pathlib import Path
import math

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

def spectrogram_array(sounds: np.ndarray, names: list[str], sample_rate: np.ndarray | None = None):
    """Plot one spectrogram (dB, linear frequency axis) per audio file in rows of 4 spectrograms

    """
    n_rows = math.ceil(len(names)/4)
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 3*n_rows), squeeze=False)
    axes = axes.flatten()
    for n, i in enumerate(sounds):
        spectrogram(i, names[n], sample_rate[n] if sample_rate is not None else None, axes[n])
    for ax in axes[len(sounds):]: ax.axis("off")
    return fig

def spectrogram(file: np.ndarray, name: str, sample_rate: int | float | None =None, axis: plt.Axes | None =None,):
    """Plot and return the spectrogram (dB, linear frequency axis) of a single signal."""
    config = open_config()
    if axis is None:
        fig, ax = plt.subplots()
    else:
        ax = axis
        fig = axis.figure
    if sample_rate is None:
        sample_rate = config["sample-rate"]
    # Convert amplitude to dB so quiet components remain visible
    spec = lib.amplitude_to_db(np.abs(lib.stft(file, n_fft=config['n-fft'])), ref=np.max)
    # hop_length must match the STFT, which uses librosa's default (n_fft // 4)
    img = lib.display.specshow(spec, sr=sample_rate, hop_length=config['n-fft'] // 4,
                            x_axis="time", y_axis="linear", ax=ax)

    ax.set_title(f"{name}, sr={sample_rate}")
    return fig

def open_config() -> dict[str, Any]:
    """Load config/config.toml and return it as a dict."""
    # Path built from this file's location, so it works from any working directory
    CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.toml"

    # tomllib requires the file to be opened in binary mode
    with open(CONFIG_PATH, "rb") as f:
        config = tomllib.load(f)
    return config
