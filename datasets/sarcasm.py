import os
import nltk

import utils
from .base import Dataset


class Sarcasm(Dataset):
    dataset_url = "http://nldslab.soe.ucsc.edu/sarcasm/sarcasm_v2.zip"

    def load_data(self):
        dataset_dir = os.path.join(self.data_dir, 'sarcasm_v2')
        cls_id = {"notsarc": 0, "sarc": 1}

        self.data = [[], []]

        nltk.download('punkt')
        for file_name in ["GEN-sarc-notsarc.csv", "HYP-sarc-notsarc.csv", "RQ-sarc-notsarc.csv"]:
            file_path = os.path.join(dataset_dir, file_name)
            items = utils.load_from_csv(file_path)
            for item in items:
                cls, idx, doc = item["class"], item["id"], item["text"]
                tokenized = nltk.word_tokenize(doc)
                self.data[cls_id[cls]].append(tokenized)
