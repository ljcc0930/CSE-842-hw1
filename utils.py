import os
import wget
import tarfile


def ensure_download_data(url, name, dir):
    file_path = os.path.join(dir, name)
    if not os.path.exists(file_path):
        os.makedirs(dir, exist_ok=True)
        wget.download(url=url, out=dir)

        file = tarfile.open(file_path)
        file.extractall(dir)


def load_from_txt(path):
    fin = open(path, 'r')
    lines = fin.readlines()
    return lines