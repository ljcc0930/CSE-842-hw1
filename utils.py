import os
import wget
import tarfile
import zipfile
import itertools
import numpy as np
import pickle as pkl
import json
import csv
import random

# import torch


def set_seed(seed):
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def evaluation(output, target):
    output = np.array(output)
    target = np.array(target)

    true_positive = np.sum(output * target)
    false_positive = np.sum(output * (1 - target))
    false_negative = np.sum((1 - output) * target)
    true_negative = np.sum((1 - output) * (1 - target))

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1 = 2 * (precision * recall) / (precision + recall)
    accuracy = (true_positive + true_negative) / \
        (true_positive + false_positive + true_negative + false_negative)

    return precision, recall, f1, accuracy


def unzip_file(path: str, out_dir):
    if path.endswith(('.tar.gz', '.tar')):
        with tarfile.open(path) as file:
            file.extractall(out_dir)
    elif path.endswith('.zip'):
        with zipfile.ZipFile(path, 'r') as file:
            file.extractall(out_dir)


def ensure_download_data(url, dir, name=None):
    file_path = os.path.join(dir, name)
    if not os.path.exists(file_path):
        os.makedirs(dir, exist_ok=True)
        wget.download(url=url, out=dir)
        unzip_file(file_path, dir)


def load_from_csv(path):
    items = []
    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            items.append(row)
    return items


def load_from_txt(path):
    fin = open(path, 'r')
    lines = fin.readlines()
    return [line.split(' ')[:-1] for line in lines]


def concat_lists(*args):
    return list(itertools.chain(*args))


def dump_obj(obj, path, method='pickle'):
    if method == "pickle":
        pkl.dump(obj, open(path, 'wb'))
    elif method == "json":
        json.dump(obj, open(path, 'w'))
    # elif method == "torch":
    #     torch.save(obj, path)
    else:
        raise NotImplementedError("Dump method {} is not implement!")


def load_obj(path, method='pickle'):
    if method == "pickle":
        return pkl.load(open(path, 'wb'))
    elif method == "json":
        return json.load(open(path, 'w'))
    # elif method == "torch":
    #     return torch.load(path)
    else:
        raise NotImplementedError("Load method {} is not implement!")
