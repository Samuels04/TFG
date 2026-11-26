from typing import Any

import numpy as np
import scipy.signal as sig

def hilbert(signal: np.ndarray, config: dict[str, Any]):
    raw_envelope = np.abs(sig.hilbert(signal))
    butter_filter = sig.butter(3, config["hilbert-cutoff"], btype='lowpass', fs=config["sample-rate"], output="sos")
    res = sig.sosfiltfilt(butter_filter, raw_envelope)
    return res

def shannon(signal: np.ndarray, config: dict[str, Any]):
    non_zero_constant = 1e-12
    #First we normalize the signal
    signal_norm = signal / (np.max(np.abs(signal)) + non_zero_constant)
    #Then we obtain the envelope from that normalized signal
    res = (signal_norm**2 * -1)*np.log(signal_norm**2 + non_zero_constant)
    return res 