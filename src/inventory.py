from pathlib import Path
import pandas as pd
import soundfile as sf
import numpy as np
import src.io
import re
import hashlib
import yaml
from sklearn.model_selection import StratifiedGroupKFold

def generate():
    """Generates the inventory.csv file containing data from all audio files present accross all available datasets. 
    Returns the inventory DataFrame"""
    inventory_l = []
    # Open the global configuration file
    config = src.io.open_config()
    # Raw data directory to traverse
    raw_data = config["raw-data"]
    # Bit depth map for finding the adequate bit depth for the file's subtype
    subtype_bit_depth: dict[str, pd.Int64Dtype] = {'PCM_16': 16, 'PCM_24': 24, 'PCM_32': 32, 'PCM_U8': 8, 'FLOAT': 32, 'DOUBLE': 64, 'ULAW': None, 'ALAW': None, 'IMA_ADPCM': None, 'MS_ADPCM': None, 
                         'GSM610': None, 'G721_32': None, 'NMS_ADPCM_16': None, 'NMS_ADPCM_24': None, 'NMS_ADPCM_32': None, 'MPEG_LAYER_III': None}

    # Traverse the whole raw_data directory and check all .wav files
    for file in Path(raw_data).rglob("*.wav"):

        #Extract the dataset and subfolder from the file's path
        dataset = file.relative_to(raw_data).parts[0]
        folder = file.relative_to(raw_data).parts[1]

        # Read the annotations according to each dataset, if the dataset is not found, an error is raised
        match dataset:
            case "HLS-CMDS":
                patient, location, label = src.datasets.hls.read_annotations(file, folder)
                if label is None:
                    lung_label = None
                    heart_label = None
                    strat_key = "HLS:unlabelled"
                else:
                    lung_label = label[0]
                    heart_label = label[1]
                    if pd.isna(lung_label):
                        strat_key = f"HLS-H:{"Normal" if heart_label == "Normal" else "Abnormal"}"
                    elif pd.isna(heart_label):
                        f"HLS-L:{"Normal" if lung_label == "Normal" else "Abnormal"}"
                    elif pd.isna(lung_label) and pd.isna(heart_label):
                        strat_key = "HLS:unlabelled"
                    else:
                        strat_key = f"HLS-L:{"Normal" if lung_label == "Normal" else "Abnormal"}|H:{"Normal" if heart_label == "Normal" else "Abnormal"}"
                disease = None
            case "NIH":
                patient, location, disease = src.datasets.nih.read_annotations(file)
                lung_label = None
                strat_key = "NIH:unlabelled" if pd.isna(disease) else f"NIH:{disease}"
            case _:
                raise ValueError()

        # Signal checks, not in the file's metadata
        audio, sr = sf.read(file)
        peak = np.max(np.abs(audio))
        rms = np.sqrt(np.mean(audio**2))
        dc_offset = np.mean(audio)
        clipping = np.mean(np.abs(audio) >= 0.999)
        # Retrieve the file's metadata from its header
        info = sf.info(file)
        sr_f = info.samplerate
        duration = info.duration
        channels = info.channels
        subtype = info.subtype
        bit_depth = subtype_bit_depth[subtype]
        is_float = subtype in ('FLOAT', 'DOUBLE')
        # Generate the file's hash value with the sha-256 algorithm
        with file.open("rb") as f: 
            hash = hashlib.file_digest(f, "sha256").hexdigest()
        # Append the row with the values to a list of lists
        inventory_l.append([
            file.name, dataset, patient, location, hash, sr_f, channels, bit_depth, is_float, peak, rms, dc_offset, clipping, duration, lung_label, heart_label, disease, strat_key
        ])
    #From the list of lists created during the traversal, generate a DataFrame, which will be parsed into a .csv file and then returned
    inventory = pd.DataFrame(inventory_l, columns=["file", "dataset", "patient", "position", "hash", "sample_rate", "channels", "bit_depth", "is_float", "peak", "rms", "dc_offset", "clipping", "duration", "lung_label", "heart_label", "disease", "strat_key"])
    inventory.to_csv("../data/interim/inventory.csv", index=False)
    print("Inventory file has been generated")
    return inventory

def merge_duplicate_groups(inventory: pd.DataFrame):
    """Adds a column 'group' to the inventory so the patiens that share any audio file (have same hash)
    get the same ID"""
    inv = inventory.copy()
    inv["group"] = inv["patient"]
    while True:
        # smallest group id among files with same hash
        canon = inv.groupby("hash")["group"].transform("min").fillna(inv["group"])
        mapping = canon.groupby(inv["group"]).min()
        new = inv["group"].map(mapping)
        if new.equals(inv["group"]):
            break
        inv["group"] = new
    return inv

def k_fold(inventory: pd.DataFrame):
    inv = inventory.copy()
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2027)
    inv["fold"] = -1
    for n, (train, test) in enumerate(sgkf.split(inv, inv["strat_key"], inv["group"])):
        inv.loc[inv.index[test], "fold"] = n

    return inv
def split(inventory: pd.DataFrame, test_fold: int = 0, val_fold: int = 1):
    test = inventory[inventory["fold"] == test_fold]
    val = inventory[inventory["fold"] == val_fold]
    train = inventory[~inventory["fold"].isin([test_fold, val_fold])]
    return train, val, test
def save(inventory: pd.DataFrame): 
    inventory.to_csv("../data/interim/inventory.csv", index=False)

def search_row(row: pd.Series):
    match row.dataset:
            case "HLS-CMDS":   
                ds="hls" 
                if pd.isna(row.lung_label) and ~pd.isna(row.heart_label):
                    type = "heart"
                elif ~pd.isna(row.lung_label) and pd.isna(row.heart_label):
                    type = "lung"
                elif ~pd.isna(row.lung_label) and ~pd.isna(row.heart_label):
                    type = "mix"
                else:
                    raise ValueError("Both labels cannot be NaN")
            case "NIH":
                ds="nih"
                type = "audio"
    return ds, type