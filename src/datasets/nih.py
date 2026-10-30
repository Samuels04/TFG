from pathlib import Path
import pandas as pd
from pandas import DataFrame
import re
PATH = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "NIH"

def read_annotations(file: Path):
    """Reads the annotations for the dataset, extracts the corresponding file's row and returns
    the patient code, the location of the recording and the label for the recording"""
    #Read the excel file
    annotations = pd.read_excel(f"{PATH}/Data annotation.xlsx")
    #Extract the patient number from the filename and generate the patient code
    patient_no = int(re.search(r"\d+(?=_)", file.name).group())
    patient = f"NIH{patient_no}"
    #Extract the row corresponding to the patient, then from the row get the location and the label
    row = annotations.loc[patient_no - 1]
    location = row["Location"]
    label = str(row["Diagnosis"]).lower()
    return patient, location, label

def group_label(x: DataFrame):
    #TODO
    return