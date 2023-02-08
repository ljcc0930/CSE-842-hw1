import os

import utils
from .base import Dataset


class Polarity(Dataset):
    dataset_url = "https://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz"

    def load_data(self):
        dataset_dir = os.path.join(self.data_dir, 'txt_sentoken')

        dirs = [os.path.join(dataset_dir, cls)
                for cls in ['neg', 'pos']]  # 0 = neg, 1 = pos

        self.data = [[], []]

        for label, class_dir in enumerate(dirs):
            for file in os.scandir(class_dir):
                if file.is_file():
                    if file.name.endswith(".txt"):
                        doc = utils.load_from_txt(file.path)
                        doc = utils.concat_lists(*doc)
                        self.data[label].append(doc)
