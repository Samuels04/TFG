from pathlib import Path
import pandas as pd
import re
PATH = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "HLS-CMDS"

def read_annotations(file: Path, type_rec: str = "Lung"):

    #Depending on the type of data to be read, we should open the corresponding file
    match type_rec:
        case "LS": 
            annotations = pd.read_csv(f"{PATH}/LS.csv")
            soundID = re.search(r".*(?=.wav$)", file.name).group()
            row =  annotations.loc[annotations["Lung Sound ID"] == soundID]
            patient = f"HLS-LS{row.index}"
            return patient, row
        case "HS":
            annotations = pd.read_csv(f"{PATH}/HS.csv")
            soundID = re.search(r".*(?=.wav$)", file.name).group()
            row = annotations.loc[annotations["Heart Sound ID"] == soundID]
            patient = f"HLS-HS{row.index}"
            return patient, row
        case "Mix":
            annotations = pd.read_csv(f"{PATH}/Mix.csv")
            soundID = re.search(r".*(?=.wav$)", file.name).group()
            patient = f"HLS-Mix{int(re.search(r"\d*(?=.wav$)", file.name).group()) - 1}"
            row = annotations.loc[(annotations["Lung Sound ID"] == soundID) | (annotations["Heart Sound ID"] == soundID) | (annotations["Mixed Sound ID"] == soundID)]
            return patient, row

    
    print(f"file={file.name}, soundID={soundID}")
    

    