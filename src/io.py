import librosa as lib
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import yaml
from pathlib import Path

def load_dataset_csv(directory: str, annotation_file: str, subdirectory: str):
    """Load dataset (that has .csv annotations) for working with the files. Returns the annotations file,
        an array with the filenames, an array with the files and an array with the sample_rates of the files"""
    #Load annotations for HLS-CMDS
    annotations = pd.read_csv(f"{directory}/{annotation_file}")

    #Load lung sounds into array where each element is one file
    files = sorted((i for i in os.listdir(f"{directory}/{subdirectory}") if i.endswith(".wav")))
    sounds = np.empty(len(files), dtype=object)
    srl = np.empty(len(files), dtype=object)
    for n, i in enumerate(files):
        f, sr = lib.load(f"{directory}/{subdirectory}/{i}", sr=None)
        sounds[n] = f
        srl[n] = sr
    return annotations, files, sounds, srl

def load_dataset_xlsx(directory: str, annotation_file: str, subdirectory: str):
    """Load dataset (that has .csv annotations) for working with the files. Returns the annotations file,
        an array with the filenames, an array with the files and an array with the sample_rates of the files"""
    #Load annotations for HLS-CMDS
    annotations = pd.read_excel(f"{directory}/{annotation_file}")

    #Load lung sounds into array where each element is one file
    files = sorted((i for i in os.listdir(f"{directory}/{subdirectory}") if i.endswith(".wav")))
    sounds = np.empty(len(files), dtype=object)
    srl = np.empty(len(files), dtype=object)
    for n, i in enumerate(files):
        f, sr = lib.load(f"{directory}/{subdirectory}/{i}", sr=None)
        sounds[n] = f
        srl[n] = sr
    return annotations, files, sounds, srl

def spectrogram(lungSounds: np.ndarray, lungFiles: list[str], lungSR: np.ndarray):
    """Generate the spectrogram for a given audio file"""
    config = open_config()
    for n, f in enumerate(lungSounds):
        fig = plt.figure(figsize=(6,4))
        #Transform the amplitude to dB so we have a logarithmic instead of linear plot
        spec = lib.amplitude_to_db(np.abs(lib.stft(f, n_fft=config['n-fft'])), ref=np.max)
        img = lib.display.specshow(spec, sr=lungSR[n], hop_length=config['n-fft'] // 4,
                                x_axis="time", y_axis="linear")
        plt.title(f"{lungFiles[n]}, sr={lungSR[n]}")
    return fig

def open_config(): 
    """Open and return the global configuration file config.yaml"""
    # config.yaml lives in the "config" directory, a sibling of this file's directory
    CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    return config
