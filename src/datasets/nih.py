from pathlib import Path
import pandas as pd
PATH = Path(__file__).resolve().parent.parent / "data" / "NIH"

def read_annotations():

    annotations = pd.read_excel(f"{PATH}/Data annotation.xlsx")
    return annotations
