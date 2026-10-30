from pathlib import Path
import pandas as pd
import soundfile as sf
import numpy as np
import src.io
import re
import hashlib
import yaml

def generate():
    """Generates the inventory.csv file containing data from all audio files present accross all available datasets. 
    Returns the inventory DataFrame"""
    inventory_l = []
    #Open the global configuration file
    config = src.io.open_config()
    #print(sys.path)
    #Raw data directory to traverse
    raw_data = config["raw-data"]
    #Bit depth map for finding the adequate bit depth for the file's subtype
    subtype_bit_depth: dict[str, pd.Int64Dtype] = {'PCM_16': 16, 'PCM_24': 24, 'PCM_32': 32, 'PCM_U8': 8, 'FLOAT': 32, 'DOUBLE': 64, 'ULAW': None, 'ALAW': None, 'IMA_ADPCM': None, 'MS_ADPCM': None, 
                         'GSM610': None, 'G721_32': None, 'NMS_ADPCM_16': None, 'NMS_ADPCM_24': None, 'NMS_ADPCM_32': None, 'MPEG_LAYER_III': None}

    #Traverse the whole raw_data directory and check all .wav files
    for file in Path(raw_data).rglob("*.wav"):

        #Extract the dataset and subfolder from the file's path
        dataset = file.relative_to(raw_data).parts[0]
        folder = file.relative_to(raw_data).parts[1]
        #print(folder)
        #Read the annotations according to each dataset, if the dataset is not found, an error is raised
        match dataset:
            case "HLS-CMDS":
                patient, location, label = src.datasets.hls.read_annotations(file, folder)
                lung_label = None if label is None else label[0]
                heart_label = None if label is None else label[1]
                disease = None
            case "NIH":
                patient, location, disease = src.datasets.nih.read_annotations(file)
                lung_label = None
            case _:
                raise ValueError()

        #display(annotations)
        ##Signal checks, not in the file's metadata
        audio, sr = sf.read(file)
        peak = np.max(np.abs(audio))
        rms = np.sqrt(np.mean(np.mean(audio**2)))
        dc_offset = np.mean(audio)
        clipping = np.mean(np.abs(audio) >= 0.999)
        #Retrievet the file's metadata from its header
        info = sf.info(file)
        sr_f = info.samplerate
        duration = info.duration
        channels = info.channels
        fmt = info.format_info
        #extra_info = info.extra_info
        subtype = info.subtype
        bit_depth = subtype_bit_depth[subtype]
        is_float = subtype in ('FLOAT', 'DOUBLE')
        #Generate the file's hash value with the sha-256 algorithm
        with file.open("rb") as f: 
            hash = hashlib.file_digest(f, "sha256").hexdigest()
        #Append the row with the values to a list of lists
        inventory_l.append([
            file.name, dataset, patient, location, hash, sr_f, channels, bit_depth, is_float, peak, rms, dc_offset, clipping, duration, lung_label, heart_label, disease
        ])
        #break
    #From the list of lists created during the traversal, generate a DataFrame, which will be parsed into a .csv file and then returned
    inventory = pd.DataFrame(inventory_l, columns=["file", "dataset", "patient", "position", "hash", "sample_rate", "channels", "bit_depth", "is_float", "peak", "rms", "dc_offset", "clipping", "duration", "lung_label", "heart_label", "disease"])
    inventory.to_csv("../data/interim/inventory.csv")
    print("Inventory file has been generated")
    return inventory