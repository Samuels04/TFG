from pathlib import Path
import pandas as pd
import soundfile as sf
import numpy as np
import src.io
import re
import hashlib
from enum import Enum
from sklearn.model_selection import StratifiedGroupKFold
from src.types import AudioType

def generate():
    """Build the inventory: one row per .wav file across all datasets in raw-data.

    Each row holds the file's location (path relative to raw-data), its annotations (patient,
    position, labels, strat_key) and content (AudioType.to_char()), its header
    metadata (sample rate, channels, bit depth, duration), simple signal checks (peak, RMS,
    DC offset, clipping) and its SHA-256 hash.

    Writes the result to ../data/interim/inventory.csv (relative to the working directory,
    so it must be run from notebooks/) and returns it as a DataFrame.
f
    Raises:
        ValueError: if a file belongs to a dataset with no adapter.
    """
    inventory_l = []
    config = src.io.open_config()
    raw_data = config["raw-data"]
    # soundfile only reports the subtype; map it to bit depth (None for compressed formats)
    subtype_bit_depth: dict[str, pd.Int64Dtype] = {'PCM_16': 16, 'PCM_24': 24, 'PCM_32': 32, 'PCM_U8': 8, 'FLOAT': 32, 'DOUBLE': 64, 'ULAW': None, 'ALAW': None, 'IMA_ADPCM': None, 'MS_ADPCM': None,
                         'GSM610': None, 'G721_32': None, 'NMS_ADPCM_16': None, 'NMS_ADPCM_24': None, 'NMS_ADPCM_32': None, 'MPEG_LAYER_III': None}

    for file in Path(raw_data).rglob("*.wav"):

        # Path is raw-data/<dataset>/<sub-folder>/...
        # Stored in the inventory so the file can be found again without deducing its folder
        path = file.relative_to(raw_data).as_posix()
        dataset = file.relative_to(raw_data).parts[0]
        audio_type = AudioType.match_file(dataset, file.relative_to(raw_data).parts[1])

        # Each dataset has its own adapter in src/datasets
        match dataset:
            case "HLS-CMDS":
                patient, location, label, strat_key, content = src.datasets.hls.read_annotations(file, audio_type)
                if label is None:
                    lung_label = None
                    heart_label = None
                else:
                    lung_label = label[0]
                    heart_label = label[1]
                disease = None
            case "NIH":
                patient, location, disease, strat_key, content = src.datasets.nih.read_annotations(file)
                lung_label = None
                heart_label = None
            case _:
                raise ValueError()

        # Signal checks computed from the samples (not available in the header).
        # Clipping = fraction of samples at full scale.
        audio, sr = sf.read(file)
        peak = np.max(np.abs(audio))
        rms = np.sqrt(np.mean(audio**2))
        dc_offset = np.mean(audio)
        clipping = np.mean(np.abs(audio) >= 0.999)
        info = sf.info(file)
        sr_f = info.samplerate
        duration = info.duration
        channels = info.channels
        subtype = info.subtype
        bit_depth = subtype_bit_depth[subtype]
        is_float = subtype in ('FLOAT', 'DOUBLE')
        # The hash detects identical audio stored under different names (used by merge_duplicate_groups)
        with file.open("rb") as f:
            hash = hashlib.file_digest(f, "sha256").hexdigest()
        inventory_l.append([
            file.name, path, dataset, patient, location, hash, sr_f, channels, bit_depth, is_float, peak, rms, dc_offset, clipping, duration, lung_label, heart_label, disease, strat_key, content
        ])
    inventory = pd.DataFrame(inventory_l, columns=["file", "path", "dataset", "patient", "position", "hash", "sample_rate", "channels", "bit_depth", "is_float", "peak", "rms", "dc_offset", "clipping", "duration", "lung_label", "heart_label", "disease", "strat_key", "content"])
    inventory.to_csv("../data/interim/inventory.csv", index=False)
    print("Inventory file has been generated")
    return inventory

def merge_duplicate_groups(inventory: pd.DataFrame):
    """Add a 'group' column so that patients sharing any identical audio file end up in the same group.

    The group is what the split uses, so identical audio can never be in two different subsets.
    Returns a copy of the inventory; the input is not modified.
    """
    inv = inventory.copy()
    inv["group"] = inv["patient"]
    # Repeat until nothing changes so the merge is transitive:
    # if A shares a file with B and B with C, then A, B and C all end up in the same group
    while True:
        # Each hash takes the smallest group ID among its files...
        canon = inv.groupby("hash")["group"].transform("min").fillna(inv["group"])
        # ...and each group takes the smallest of those IDs over all its files
        mapping = canon.groupby(inv["group"]).min()
        new = inv["group"].map(mapping)
        if new.equals(inv["group"]):
            break
        inv["group"] = new
    return inv

def k_fold(inventory: pd.DataFrame):
    """Assign each row to one of 5 folds (new 'fold' column, 0-4).

    Uses StratifiedGroupKFold: all rows of a group go to the same fold, and folds keep a
    similar proportion of each strat_key. Requires the 'group' and 'strat_key' columns.
    Returns a copy of the inventory.
    """
    inv = inventory.copy()
    # Fixed random_state so the folds are reproducible
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2027)
    # -1 = not assigned; any -1 left after the loop means something went wrong
    inv["fold"] = -1
    for n, (train, test) in enumerate(sgkf.split(inv, inv["strat_key"], inv["group"])):
        inv.loc[inv.index[test], "fold"] = n

    return inv
def split(inventory: pd.DataFrame, test_fold: int = 0, val_fold: int = 1):
    """Split the inventory into (train, val, test) DataFrames using the 'fold' column.

    test_fold and val_fold go to test and validation; every other fold goes to train.
    """
    test = inventory[inventory["fold"] == test_fold]
    val = inventory[inventory["fold"] == val_fold]
    train = inventory[~inventory["fold"].isin([test_fold, val_fold])]
    return train, val, test
def save(inventory: pd.DataFrame):
    """Overwrite ../data/interim/inventory.csv (relative to the working directory) with the given inventory."""
    inventory.to_csv("../data/interim/inventory.csv", index=False)

