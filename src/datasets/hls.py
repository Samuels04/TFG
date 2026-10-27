from pathlib import Path
import pandas as pd
PATH = Path(__file__).resolve().parent.parent / "data" / "HLS-CMDS"

def read_annotations(file: Path, dtype: str = "Lung"):

    #Depending on the type of data to be read, we should open the corresponding file
    match dtype:
        case "Lung": 
            annotations = pd.read_csv(f"{PATH}/LS.csv")
        case "Heart":
            annotations = pd.read_csv(f"{PATH}/HS.csv")
        case "Mixed":
            annotations = pd.read_csv(f"{PATH}/Mix.csv")

    return annotations.loc(file.name)

    