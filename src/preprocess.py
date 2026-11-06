import librosa as lib, math, scipy.signal as sig, src, pandas as pd


def resample(row: pd.Series, target_sample_rate: int, config):
    dataset, group = src.inventory.search_row(row)
    file, sr = lib.load(f"{config[dataset][group]}/{row.file}", sr=None)
    gcd = math.gcd(target_sample_rate, sr)
    file = sig.resample_poly(file, target_sample_rate//gcd, sr//gcd)
    return file