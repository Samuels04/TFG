import librosa as lib
import numpy as np
import scipy.signal as sig
import math
import src
from src.types import AudioType


def resample(file: np.ndarray, sr: int | float, target_sample_rate: int, config):
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
    # NOTE: debug prints
    print(f"post: {rms}")
    normalized_file /= (rms + EPSILON)
    rms = np.sqrt(np.mean(normalized_file**2))
    print(f"post-post: {rms}")

    return normalized_file

def band_split(file: np.ndarray, type: AudioType, config, target_sr):
    match type:
        case AudioType.LUNG:
            window = [config["lung-window"]]
            sos = sig.butter(N=4, Wn=window, btype='bandpass', output="sos", fs=target_sr)
        case AudioType.HEART:
            sos = sig.butter(N=4, Wn=window, btype='bandpass', output="sos", fs=target_sr)
        case AudioType.MIX:
            sos = sig.butter(N=4, Wn=window, btype='bandpass', output="sos", fs=target_sr)
        case AudioType.UNKNOWN:
            raise ValueError("Unknown type of audio file")
    return file