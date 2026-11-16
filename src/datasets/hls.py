from pathlib import Path
import pandas as pd
from pandas import DataFrame
import re
from src.types import AudioType
PATH = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "HLS-CMDS"

def read_annotations(file: Path, type_rec: AudioType = AudioType.LUNG):
    """Reads the annotations for the dataset, extracts the corresponding file's row and returns
    the patient code, the location of the recording, the label for the recording and its stratification
    key, all depending on the type of audio file received"""
    #Depending on the type of data to be read, we should open the corresponding file
    match type_rec:
        case AudioType.LUNG: 
            # Read the corresponding csv file
            annotations = pd.read_csv(f"{PATH}/LS.csv")
            annotations["Lung Sound ID"] = annotations["Lung Sound ID"].str.replace("_G_", "_CC_")
            annotations["Lung Sound ID"] = annotations["Lung Sound ID"].str.replace("_C_", "_FC_")
            #Take out the .wav from the file to get the sound ID
            soundID = re.search(r".*(?=.wav$)", file.name).group()
            #Get the row that contains the data for the file by matching the soundID
            row =  annotations.loc[annotations["Lung Sound ID"] == soundID]
            #The label is a tuple with the type of sound, the first value is Lung and second is Heart
            if row["Lung Sound Type"].empty or pd.isna(row["Lung Sound Type"].values[0]):
                label = None
            else:
                label = (str(row["Lung Sound Type"].values[0]), None)
            #The patient is the dataset-type_rec-index of the recording
            patient = f"HLS-LS{row.index.start}"
            #The location of the recording
            location = None if row["Location"].empty else row["Location"].values[0]
            if label is None:
                strat_key = f"HLS-L:unlabelled"
            else:
                strat_key = f"HLS-L:{"Normal" if label[0] == "Normal" else "Abnormal"}"
            return patient, location, label, strat_key
        case AudioType.HEART:
            # Read the corresponding csv file
            annotations = pd.read_csv(f"{PATH}/HS.csv")
            #Take out the .wav from the file to get the sound ID
            soundID = re.search(r".*(?=.wav$)", file.name).group()
            #Get the row that contains the data for the file by matching the soundID
            row = annotations.loc[annotations["Heart Sound ID"] == soundID]
            #The label is a tuple with the type of sound, the first value is Lung and second is Heart
            if row["Heart Sound Type"].empty or pd.isna(row["Heart Sound Type"].values[0]):
                label = None
            else:
                label = (None, row["Heart Sound Type"].values[0])
            #The patient is the dataset-type_rec-index of the recording
            patient = f"HLS-HS{row.index.start}"
            #The location of the recording
            location = None if row["Location"].empty else row["Location"].values[0]
            if label is None:
                strat_key = f"HLS-H:unlabelled"
            else:
                strat_key = f"HLS-H:{"Normal" if label[1] == "Normal" else "Abnormal"}"
            return patient, location, label, strat_key
        case AudioType.MIX:
            # Read the corresponding csv file
            annotations = pd.read_csv(f"{PATH}/Mix.csv")
            #Take out the .wav from the file to get the sound ID
            soundID = re.search(r".*(?=.wav$)", file.name).group()
            #The patient is the dataset-type_rec-index of the recording
            patient = f"HLS-Mix{int(re.search(r"\d*(?=.wav$)", file.name).group()) - 1}"
            #Get the row that contains the data for the file by matching the soundID with the Heart, Lung or Mixed ID in the table
            row = annotations.loc[(annotations["Lung Sound ID"] == soundID) | (annotations["Heart Sound ID"] == soundID) | (annotations["Mixed Sound ID"] == soundID)]
            #The label is a tuple with the type of sound, the first value is Lung and second is Heart
            if row["Heart Sound Type"].empty or pd.isna(row["Heart Sound Type"].values[0]):
                labelH = None
            else:
                labelH = str(row["Heart Sound Type"].values[0])
                
            if row["Lung Sound Type"].empty or pd.isna(row["Lung Sound Type"].values[0]):
                labelL = None
            else:
                labelL = str(row["Lung Sound Type"].values[0])

            label = None if (labelH is None or labelL is None) else (labelL, labelH)
            #The location of the recording
            location = None if row["Location"].empty else row["Location"].values[0]
            if label is None:
                strat_key = f"HLS:unlabelled"
            else:
                strat_key = f"HLS-L:{"Normal" if label[0] == "Normal" else "Abnormal"}|H:{"Normal" if label[1] == "Normal" else "Abnormal"}"
            return patient, location, label, strat_key
        case _: 
            raise ValueError(f"... {file}")