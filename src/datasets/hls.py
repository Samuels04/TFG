from pathlib import Path
import pandas as pd
import re
PATH = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "HLS-CMDS"

def read_annotations(file: Path, type_rec: str = "Lung"):

    #Depending on the type of data to be read, we should open the corresponding file
    match type_rec:
        case "LS": 
            # Read the corresponding csv file
            annotations = pd.read_csv(f"{PATH}/LS.csv")
            annotations["Lung Sound ID"] = annotations["Lung Sound ID"].str.replace("_G_", "_CC_")
            annotations["Lung Sound ID"] = annotations["Lung Sound ID"].str.replace("_C_", "_FC_")
            #Take out the .wav from the file to get the sound ID
            soundID = re.search(r".*(?=.wav$)", file.name).group()
            #Get the row that contains the data for the file by matching the soundID
            row =  annotations.loc[annotations["Lung Sound ID"] == soundID]
            #The label is a tuple with the type of sound, the first value is Lung and second is Heart
            label = None if row["Lung Sound Type"].empty else (str(row["Lung Sound Type"].values[0]), None)
            #The patient is the dataset-type_rec-index of the recording
            patient = f"HLS-LS{row.index.start}"
            return patient, row, label
        case "HS":
            # Read the corresponding csv file
            annotations = pd.read_csv(f"{PATH}/HS.csv")
            #Take out the .wav from the file to get the sound ID
            soundID = re.search(r".*(?=.wav$)", file.name).group()
            #Get the row that contains the data for the file by matching the soundID
            row = annotations.loc[annotations["Heart Sound ID"] == soundID]
            #The label is a tuple with the type of sound, the first value is Lung and second is Heart
            label = None if row["Heart Sound Type"].empty else (None, str(row["Heart Sound Type"].values[0]))
            #The patient is the dataset-type_rec-index of the recording
            patient = f"HLS-HS{row.index.start}"
            return patient, row, label
        case "Mix":
            # Read the corresponding csv file
            annotations = pd.read_csv(f"{PATH}/Mix.csv")
            #Take out the .wav from the file to get the sound ID
            soundID = re.search(r".*(?=.wav$)", file.name).group()
            #The patient is the dataset-type_rec-index of the recording
            patient = f"HLS-Mix{int(re.search(r"\d*(?=.wav$)", file.name).group()) - 1}"
            #Get the row that contains the data for the file by matching the soundID with the Heart, Lung or Mixed ID in the table
            row = annotations.loc[(annotations["Lung Sound ID"] == soundID) | (annotations["Heart Sound ID"] == soundID) | (annotations["Mixed Sound ID"] == soundID)]
            #The label is a tuple with the type of sound, the first value is Lung and second is Heart
            labelH = None if row["Heart Sound Type"].empty else str(row["Heart Sound Type"].values[0])
            labelL = None if row["Lung Sound Type"].empty else str(row["Lung Sound Type"].values[0])
            label = None if (labelH is None or labelL is None) else (labelL, labelH)
            return patient, row, label

    
    print(f"file={file.name}, soundID={soundID}")
    

    