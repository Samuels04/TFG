import librosa as lib
import numpy as np
import scipy.signal as sig
import math
import src
from src.inventory import Type


def resample(file: np.ndarray, sr: int | float, target_sample_rate: int, config):
    """Returns a resampled version of the signal. This signal has a sample rate equal to the target passed as an argument"""
    # First we pre-calculate the GCD of the current sample rate and the target sample rate
    gcd = math.gcd(target_sample_rate, sr)
    # Then we resample it passing the arguments as required so the resulting sample rate equals the target
    resampled_file = sig.resample_poly(file, target_sample_rate//gcd, sr//gcd)
    return resampled_file

def normalize(file: np.ndarray): 
    """Returns an RMS-Normalized version of the signal"""
    # A constant added to the RMS so there never is a division by 0
    EPSILON = 1e-8
    # First we substract from each signal (file) its mean
    normalized_file = file - np.mean(file)
    # Then we calculate its RMS
    rms = np.sqrt(np.mean(normalized_file**2))
    print(f"post: {rms}")
    # Then we divide the signal (file) by its RMS added to EPSILON just in case it ever were 0
    normalized_file /= (rms + EPSILON)
    rms = np.sqrt(np.mean(normalized_file**2))
    print(f"post-post: {rms}")

    return normalized_file

def band_split(file: np.ndarray, type: Type, config, target_sr):
    match type:
        case Type.LUNG:
            sig.butter(N=4, Wn=[50, 2500], btype='bandpass', output="sos", fs=target_sr)
        case Type.HEART:
            sig.butter(N=4, Wn=[20, 400], btype='bandpass', output="sos", fs=target_sr)
        case Type.MIX:
            sig.butter(N=4, Wn=[], btype='bandpass', output="sos", fs=target_sr)
        case Type.UNKNOWN:
            raise ValueError("Unknown type of audio file")
    return file