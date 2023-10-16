''' Functions for reading tfrecords used for training data.

File copied from [Name of repo] and refactored for use reading 
simulated data produced by that project.

'''


import matplotlib.pyplot as plt
import tensorflow as tf


# feature type definitions
def _int_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# write function
def convert_tfrecords(fase, magnitud, labels, name):
    rows = fase.shape[0]
    cols = fase.shape[1]
    depth = fase.shape[2]

    fase_raw = fase.astype("float32").tostring()
    mag_raw = magnitud.astype("float32").tostring()
    label_raw = labels.astype("float32").tostring()

    filename = name + ".tfrecords"
    with tf.io.TFRecordWriter(filename) as writer:
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "height": _int_feature(rows),
                    "width": _int_feature(cols),
                    "depth": _int_feature(depth),
                    "label": _bytes_feature(label_raw),
                    "pha_raw": _bytes_feature(fase_raw),
                    "mag_raw": _bytes_feature(mag_raw),
                }
            )
        )
        writer.write(example.SerializeToString())


# read and decode functions
def decode_data(coded_data, shape):
    with tf.device("cpu"):
        data = tf.io.decode_raw(
            coded_data, 
            tf.float32, 
            fixed_length=tf.cast(4*shape[0]*shape[1]*shape[2], tf.int32)
        )
        return tf.reshape(data, shape)
    
def _parse_image_function(record_bytes, shape, include_mag: bool = False):
    with tf.device("cpu"):
        features = tf.io.parse_single_example(
            record_bytes, 
            {
                "height": tf.io.FixedLenFeature([], tf.int64),
                "width": tf.io.FixedLenFeature([], tf.int64),
                "depth": tf.io.FixedLenFeature([], tf.int64),
                "label": tf.io.FixedLenFeature([], tf.string),
                "pha_raw": tf.io.FixedLenFeature([], tf.string),
                "mag_raw": tf.io.FixedLenFeature([], tf.string),
            }
        )
        decoded = {
            'phase': tf.io.decode_raw(
                features['pha_raw'], 
                tf.float32,
                fixed_length=4*shape[0]*shape[1]*shape[2]
            ),
            'label': tf.io.decode_raw(
                features['label'], 
                tf.float32,
                fixed_length=4*shape[0]*shape[1]*shape[2]
            ),
        }
        if include_mag:
            decoded['magnitude'] = tf.io.decode_raw(
                features['mag_raw'],
                tf.float32,
                fixed_length=4*shape[0]*shape[1]*shape[2]
            )
            return (
                tf.reshape(decoded['magnitude'], shape, name='magnitude'),
                tf.reshape(decoded['phase'], shape, name='phase'),
                tf.reshape(decoded['label'], shape, name='label')
            )
        return (
            tf.reshape(decoded['phase'], shape, name='phase'),
            tf.reshape(decoded['label'], shape, name='label')
        )

def read_and_decode(filenames: list[str], shape: list[int], *, include_mag: bool = False):
    with tf.device("cpu"):
        return tf.data.TFRecordDataset(filenames).map(lambda x: _parse_image_function(x, shape, include_mag))

def show_example(
    filenames: list[str],
    shape: list[int],
    centre: list[int],
    n: int = 1,
):
    dataset = read_and_decode(filenames, shape, include_mag=True).take(n)

    for _ in range(n):
        mag, pha, sus = next(iter(dataset))

        fig, axs = plt.subplots(2, 2)

def asc_plot(img, point):
    fig, axs = plt.subplots(2, 2)

    axs[0, 0] = plt.imshow(img[:, :, point[2]])
    axs[0, 1] = plt.imshow(img[point[0], :, :])
    axs[1, 0] = plt.imshow(img[:, point[1], :])
