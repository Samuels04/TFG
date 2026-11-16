from pathlib import Path
import pandas as pd
from pandas import DataFrame
import re
from src.types import AudioType
PATH = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "NIH"

def read_annotations(file: Path):
    """Read the NIH annotations for one recording.

    Args:
        file: path to the .wav file. Its name must contain the patient number followed by "_".

    Returns:
        (patient, location, disease, strat_key, content):
        - patient: "NIH{number}".
        - location: chest position as written in the Excel file.
        - disease: diagnosis in lowercase.
        - strat_key: "NIH:{disease}", or "NIH:unlabelled" if the diagnosis is empty.
        - content: 'L' since in this dataset all files are lung files
    """
    annotations = pd.read_excel(f"{PATH}/Data annotation.xlsx")
    patient_no = int(re.search(r"\d+(?=_)", file.name).group())
    patient = f"NIH{patient_no}"
    # Assumes the Excel has one row per patient, sorted by patient number starting at 1
    row = annotations.loc[patient_no - 1]
    location = row["Location"]
    disease = row["Diagnosis"]
    # Check NaN before converting to str: str(NaN) is "nan", which pd.isna no longer detects
    strat_key = "NIH:unlabelled" if pd.isna(disease) else f"NIH:{str(disease).lower()}"
    content = AudioType.LUNG.value
    return patient, location, str(disease).lower(), strat_key, content