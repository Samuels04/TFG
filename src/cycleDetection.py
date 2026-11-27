from typing import Any

import numpy as np
import scipy.signal as sig
from scipy.ndimage import uniform_filter1d

def hilbert(signal: np.ndarray, sample_rate, cutoff):
    raw_envelope = np.abs(sig.hilbert(signal))
    butter_filter = sig.butter(3, cutoff, btype='lowpass', fs=sample_rate, output="sos")
    res = sig.sosfiltfilt(butter_filter, raw_envelope)
    return res

def shannon(signal: np.ndarray, sample_rate, window_size: float):
    non_zero_constant = 1e-12
    #First we normalize the signal
    p = np.percentile(np.abs(signal), 99)
    signal_norm = np.clip(signal / (p + non_zero_constant), -1, 1)
    #Then we obtain the envelope from that normalized signal
    env = (signal_norm**2 * -1)*np.log(signal_norm**2 + non_zero_constant)

    # Size of the window of the moving average in samples
    size = int(window_size * sample_rate)
    # Make it odd so we have a perfectly symmetric window
    size = size | 1
    print(size)
    # Calculation of the moving average. I choose the constant mode as I assume silence 
    # beyond the edges of the recording, so as not to introduce any fake data in the future
    res = uniform_filter1d(env, size, mode = "constant")
    return res 