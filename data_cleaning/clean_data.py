"""

"""


from tqdm import tqdm
from pathlib import Path

from QSMLearn import tfrecords


def clean_dir(dirname, shape):
    datadir = Path(dirname)

    with open(datadir / 'logs.txt', 'w') as f:
        for path in tqdm(datadir.glob('*.tfrecords')):
            data = tfrecords.read_and_decode([str(path)], shape)
            try:
                phase, sus = next(iter(data))
                if not (phase.shape == sus.shape == shape):
                    path.unlink()
                    f.write(f"{str(path)} deleted: missing data\n")
                    print(f"{str(path)} deleted: missing data\n")
            except:
                path.unlink()
                f.write(f"{str(path)} deleted: missing data\n")
                print(f"{str(path)} deleted: missing data\n")
    return True

def main():
    BASE_DIR = r"D:\48data_normal4"
    SHAPE = [48, 48, 48]
    
    clean_dir(BASE_DIR + r'\test', SHAPE)
    clean_dir(BASE_DIR + r'\train', SHAPE)


if __name__ == '__main__':
    main()
