from pathlib import Path
import pandas as pd
import soundfile as sf
import numpy as np
import src.io
import re
import hashlib
import yaml



def generate():
    inventory_l = []
    config = src.io.open_config()
    #print(sys.path)
    raw_data = config["raw-data"]
    subtype_bit_depth: dict[str, pd.Int64Dtype] = {'PCM_16': 16, 'PCM_24': 24, 'PCM_32': 32, 'PCM_U8': 8, 'FLOAT': 32, 'DOUBLE': 64, 'ULAW': None, 'ALAW': None, 'IMA_ADPCM': None, 'MS_ADPCM': None, 
                         'GSM610': None, 'G721_32': None, 'NMS_ADPCM_16': None, 'NMS_ADPCM_24': None, 'NMS_ADPCM_32': None, 'MPEG_LAYER_III': None}

    for file in Path(raw_data).rglob("*.wav"):
        dataset = file.relative_to(raw_data).parts[0]
        folder = file.relative_to(raw_data).parts[1]
        #print(folder)
        match dataset:
            case "HLS-CMDS":
                patient, annotations = src.datasets.hls.read_annotations(file, folder)
                location = None if annotations.empty else annotations.Location.values[0]
            case "NIH":
                annotations = src.datasets.nih.read_annotations(file)
                location = annotations.Location
                patient = f"NIH{int(re.search(r"\d+(?=_)", file.name).group())}"

        #display(annotations)
        audio, sr = sf.read(file)
        peak = np.max(np.abs(audio))
        rms = np.sqrt(np.mean(np.mean(audio**2)))
        dc_offset = np.mean(audio)
        clipping = np.mean(np.abs(audio) >= 0.999)
        info = sf.info(file)
        sr_f = info.samplerate
        duration = info.duration
        channels = info.channels
        fmt = info.format_info
        #extra_info = info.extra_info
        subtype = info.subtype
        bit_depth = subtype_bit_depth[subtype]
        is_float = subtype in ('FLOAT', 'DOUBLE')
        with file.open("rb") as f: 
            hash = hashlib.file_digest(f, "sha256").hexdigest()

        inventory_l.append([
            file.name, dataset, patient, location, hash, sr_f, channels, bit_depth, is_float, peak, rms, dc_offset, clipping, duration, None,
        ])
        #break

    inventory = pd.DataFrame(inventory_l, columns=["file", "dataset", "patient", "position", "hash", "sample_rate", "channels", "bit_depth", "is_float", "peak", "rms", "dc_offset", "clipping", "duration", "annotation"])
    inventory.to_csv("../data/interim/inventory.csv")
    print("Inventory file has been generated")
    return inventory