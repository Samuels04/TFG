from pathlib import Path
import pandas as pd
import re
PATH = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "NIH"

def read_annotations(file: Path):

    annotations = pd.read_excel(f"{PATH}/Data annotation.xlsx")
    patient = int(re.search(r"\d+(?=_)", file.name).group())
    return annotations.loc[patient - 1]