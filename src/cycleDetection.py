from typing import Any

import numpy as np
import scipy.signal as sig

def hilbert(signal: np.ndarray, config: dict[str, Any]):
    raw_envelope = np.abs(sig.hilbert(signal))
    butter_filter = sig.butter(3, config["hilbert-cutoff"], btype='lowpass', fs=config["sample-rate"], output="sos")
    res = sig.sosfiltfilt(butter_filter, raw_envelope)
    return res

def shannon(signal: np.ndarray):
    # TODO
    return