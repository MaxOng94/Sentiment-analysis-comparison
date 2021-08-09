import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class customDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    # allows us to select examples through indexing
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(int(self.labels[idx]))
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


def read_data(fname:str, LABEL_COL,TEXT_COL,lower_case: bool=False) ->pd.DataFrame:
        """
        This function will read the textfiles.

        fname will be out of new_train_data.csv, unlabeled_data.csv and test_data.txt

        """
        try:
            df = pd.read_csv(fname, encoding = "UTF-8", usecols = ["class","comment"])
            df[LABEL_COL]= df[LABEL_COL].replace({"negative":0, "neutral":1, "positive":2})
            if lower_case:
                df[TEXT_COL]= df[TEXT_COL].str.lower()

            return df
        except (FileNotFoundError,PermissionError):

            print("No files found. Check the data directory for files.")
