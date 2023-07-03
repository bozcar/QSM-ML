"""

"""


from tqdm import tqdm
from pathlib import Path

import tensorflow as tf

from QSMLearn import tfrecords


def clean_dir(dirname):
    datadir = Path(dirname)

    if not (datadir / 'corrupt').exists():
        (datadir / 'corrupt').mkdir()

    with open(datadir / 'logs.txt', 'w') as f:
        for path in tqdm(datadir.glob('*.tfrecords')):
            data = tfrecords.read_and_decode([str(path)], [48, 48, 48])
            try:
                image, phase = next(iter(data))
            except:
                path.unlink()
                f.write(f"{str(path)} deleted: missing data\n")
    return True

def main():
    BASE_DIR = r"D:\48data_normal4"
    
    clean_dir(BASE_DIR + r'\test')
    clean_dir(BASE_DIR + r'\train')


if __name__ == '__main__':
    main()
