'''
File copied from [Name of repo] for use reading simulated data produced by that project.
'''


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
        data = tf.io.decode_raw(coded_data, tf.float32)
        data = tf.reshape(data, shape + [1])
        data = tf.cast(data, tf.float32)
        return data
    
def _parse_image_function(record_bytes):
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

        height = features["height"]
        width = features["width"]
        depth = features["depth"]

        shape = [height, width, depth]

        fase_raw = features["pha_raw"]
        mag_raw = features["mag_raw"]
        label_raw = features["label"]

        fase = decode_data(fase_raw, shape)
        magnitud = decode_data(mag_raw, shape)
        image = tf.concat([fase, magnitud], -1)

        label = decode_data(label_raw, shape)

        return {
            'image': image,
            'label': label
        }

def read_and_decode(filename):
    with tf.device("cpu"):
        return tf.data.TFRecordDataset(filename).map(_parse_image_function)
