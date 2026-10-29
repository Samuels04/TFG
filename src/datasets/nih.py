from pathlib import Path
import pandas as pd
import re
PATH = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "NIH"

def read_annotations(file: Path):

    annotations = pd.read_excel(f"{PATH}/Data annotation.xlsx")
    patient_no = int(re.search(r"\d+(?=_)", file.name).group())
    patient = f"NIH{patient_no}"
    row = annotations.loc[patient_no - 1]
    label = str(row["Diagnosis"]).lower()
    return patient, row, label