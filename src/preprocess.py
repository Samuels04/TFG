from typing import Any

import librosa as lib
import numpy as np
import scipy.signal as sig
import math
import src
from src.types import AudioType


def resample(file: np.ndarray, sr: int | float, target_sample_rate: int):
    """Return the signal resampled from sr to target_sample_rate.

    Uses polyphase resampling (scipy resample_poly), which also applies an anti-aliasing filter.
    """
    # NOTE: the config argument is not used
    # resample_poly works with an integer up/down ratio; dividing by the GCD gives the smallest one
    gcd = math.gcd(target_sample_rate, sr)
    resampled_file = sig.resample_poly(file, target_sample_rate//gcd, sr//gcd)
    return resampled_file

def normalize(file: np.ndarray): 
    """Return the signal with its mean removed and scaled to RMS = 1.

    This makes recordings with different gain comparable.
    """
    # Avoids dividing by 0 on a silent signal
    EPSILON = 1e-8
    # Remove the DC offset first so it does not count towards the RMS
    normalized_file = file - np.mean(file)
    rms = np.sqrt(np.mean(normalized_file**2))
    normalized_file /= (rms + EPSILON)
    rms = np.sqrt(np.mean(normalized_file**2))

    return normalized_file

def band_split(file: np.ndarray, butterworth_order, target_sr, lung_window: list, heart_window: list):
    """Splits an audio file into its lung and its heart bands
    
    Uses a butterworth filter together with second-order sections
    Args:
        - **file** : the audio file to split
        - **config** : the configuration for the project
        - **target_sr** : the target sample rate for the bands

    Returns:
        - **heart_filtered**: the band containing the heart sounds
        - **lung_filtered**: the band containing the lung sounds
    """
    sos_lung = sig.butter(N=butterworth_order, Wn=lung_window, btype='bandpass', output="sos", fs=target_sr)
    sos_heart = sig.butter(N=butterworth_order, Wn=heart_window, btype='bandpass', output="sos", fs=target_sr)


    lung_filtered: np.ndarray = sig.sosfiltfilt(sos_lung, file)
    heart_filtered: np.ndarray = sig.sosfiltfilt(sos_heart, file)
    return heart_filtered, lung_filtered

def pipeline(path: str, config: dict[str, Any]):
    signal, sample_rate = lib.load(path, sr=None)
    target_sr = config["sample-rate"]
    res = resample(file=signal, sr=sample_rate, target_sample_rate=target_sr)
    norm = normalize(res)
    lung_wn = [config["lung-band"]["low"], config["lung-band"]["high"]]
    heart_wn = [config["heart-band"]["low"], config["heart-band"]["high"]]
    heart, lung = band_split(norm, config["butterworth-filter-order"], target_sr, lung_wn, heart_wn)
    return heart, lung