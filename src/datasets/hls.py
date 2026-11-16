from pathlib import Path
import pandas as pd
from pandas import DataFrame
import re
from src.types import AudioType
PATH = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "HLS-CMDS"

def read_annotations(file: Path, type_rec: AudioType = AudioType.LUNG):
    """Read the HLS-CMDS annotations for one recording.

    Args:
        file: path to the .wav file.
        type_rec: sub-folder the file comes from (LUNG = LS, HEART = HS, MIX = Mix).
            It selects which CSV is read.

    Returns:
        (patient, location, label, strat_key, content):
        - patient: "HLS-LS{n}", "HLS-HS{n}" or "HLS-Mix{n}". HLS has no real patient IDs, so this is
          a per-recording ID.
        - location: chest position, or None if the file is not in the CSV.
        - label: tuple (lung_label, heart_label) with None in the part the file does not contain,
          or None if a label is missing.
        - strat_key: "HLS-L:{Normal|Abnormal}", "HLS-H:{...}", "HLS-L:{...}|H:{...}" or "...:unlabelled".
        - content: sound the file contains: 'L', 'H' or 'M'. For Mix it comes from the filename.

    Raises:
        ValueError: if type_rec is not LUNG, HEART or MIX, or if the content of a Mix file
            cannot be determined from its name.
    """
    # Sound ID = filename without ".wav"
    soundID = re.search(r".*(?=.wav$)", file.name).group()

    # Each sub-folder only differs in where to look: which CSV, which ID columns,
    # and what sound its files contain. Everything after the match is shared.
    match type_rec:
        case AudioType.LUNG:
            folder = "LS"
            annotations = pd.read_csv(f"{PATH}/LS.csv")
            # The CSV and the .wav filenames use different codes for the same sound
            # (CSV: _G_ / _C_, files: _CC_ / _FC_). Rewrite the CSV IDs so they match the filenames.
            annotations["Lung Sound ID"] = annotations["Lung Sound ID"].str.replace("_G_", "_CC_")
            annotations["Lung Sound ID"] = annotations["Lung Sound ID"].str.replace("_C_", "_FC_")
            id_columns = ["Lung Sound ID"]
            content = AudioType.LUNG
        case AudioType.HEART:
            folder = "HS"
            annotations = pd.read_csv(f"{PATH}/HS.csv")
            id_columns = ["Heart Sound ID"]
            content = AudioType.HEART
        case AudioType.MIX:
            folder = "Mix"
            annotations = pd.read_csv(f"{PATH}/Mix.csv")
            # Each Mix.csv row describes three files (lung, heart and mixed), so match against the three ID columns
            id_columns = ["Lung Sound ID", "Heart Sound ID", "Mixed Sound ID"]
            # Mix holds lung-only (L...), heart-only (H...) and mixed (M...) files; the name tells which
            content = AudioType.from_filename(file.name)
            if content == AudioType.UNKNOWN:
                raise ValueError(f"Could not find the content of {file.name} from its name")
        case _:
            raise ValueError(f"Unknown HLS sub-folder for {file.name}")

    row = annotations.loc[annotations[id_columns].eq(soundID).any(axis=1)]

    if type_rec == AudioType.MIX:
        # Mix filenames are numbered from 1; subtract 1 so the IDs start at 0 like the LS/HS ones
        patient = f"HLS-Mix{int(re.search(r"\d*(?=.wav$)", file.name).group()) - 1}"
    else:
        patient = f"HLS-{folder}{row.index.start}"

    location = None if row["Location"].empty else row["Location"].values[0]
    label, strat_key = _label_and_strat_key(content, _read_label(row, "Lung Sound Type"), _read_label(row, "Heart Sound Type"))
    # content stays an AudioType inside this function (the helpers compare against the enum);
    # it is converted to a character only here, for the inventory
    return patient, location, label, strat_key, content.value

def _read_label(row: DataFrame, column: str):
    """Return the label stored in `column` of the matched row as a str.

    Returns None if the CSV has no such column, if no row matched the file, 
    or if the cell is empty.
    """
    if column not in row or row[column].empty or pd.isna(row[column].values[0]):
        return None
    return str(row[column].values[0])

def _label_and_strat_key(content: AudioType, lung_label, heart_label):
    """Build (label, strat_key) keeping only the labels of the sounds the file contains.

    A lung-only file ignores the heart label and vice versa, so identical audio always gets the
    same label. If a label the file needs is missing, returns (None, "...:unlabelled").
    """
    match content:
        case AudioType.LUNG:
            if lung_label is None:
                return None, "HLS-L:unlabelled"
            return (lung_label, None), f"HLS-L:{_normal_or_abnormal(lung_label)}"
        case AudioType.HEART:
            if heart_label is None:
                return None, "HLS-H:unlabelled"
            return (None, heart_label), f"HLS-H:{_normal_or_abnormal(heart_label)}"
        case AudioType.MIX:
            if lung_label is None or heart_label is None:
                return None, "HLS:unlabelled"
            return (lung_label, heart_label), f"HLS-L:{_normal_or_abnormal(lung_label)}|H:{_normal_or_abnormal(heart_label)}"

def _normal_or_abnormal(label: str):
    """Collapse a specific label into the two classes used for stratification."""
    return "Normal" if label == "Normal" else "Abnormal"
