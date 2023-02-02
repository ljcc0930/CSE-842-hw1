import os

import torch
import utils

class Polarity:
    def __init__(self, num_folds, data_dir):
        dataset_dir = os.path.join(data_dir, 'txt_sentoken')
        self.get_data(dataset_dir)
        self.num_folds = num_folds

    def get_data(self, dataset_dir):
        dirs = [os.path.join(dataset_dir, cls) for cls in ['pro', 'neg']]

        data = [[], []]
        for label, class_dir in enumerate(dirs):
            for name in os.scandir(class_dir):
                if name.endswith('.txt'):
                    path = os.path.join(class_dir, name)
                    data[label] += utils.load_txt_file(path)
        self.folds = []
        


def get_dataset(dataset, num_folds = 3, data_dir = 'data'):
    if dataset == "polarity":
        return Polarity(num_folds, data_dir)

if __name__ == "__main__":
    get_dataset("polarity")