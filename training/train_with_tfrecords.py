"""

"""


def get_data(dirname: str, shape):
    from pathlib import Path
    
    from QSMLearn import tfrecords

    datadir = Path(dirname)
    datapaths = datadir.glob('*.tfrecords')
    datafilenames = [str(path) for path in datapaths]
    
    return tfrecords.read_and_decode(datafilenames, shape)


def main():
    from tensorflow import keras

    from QSMLearn import models

    # find all tfrecords files in `DATALIB`
    BASE_DIR = r"D:\48data_normal4"
    SHAPE = [48, 48, 48]

    training_dataset = get_data(BASE_DIR + r'\train', SHAPE).batch(
        10,
        drop_remainder=True
    )
    validation_dataset = get_data(BASE_DIR + r'\test', SHAPE).batch(
        10,
        drop_remainder=True
    )

    # NDI model with 200 trainable step sizes
    m = models.VariableStepNDI(
        iters=200,
        verbose = False
    )

    m.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss=keras.losses.MeanSquaredError(),
    )
    history = m.fit(
        x = training_dataset,
        validation_data=validation_dataset,
        batch_size = 10,
        epochs = 5,
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience = 500,
                monitor = 'val_loss',
                verbose = 1
            ),
            keras.callbacks.ReduceLROnPlateau(
                factor = 0.75, 
                patience = 25,
                min_lr = 0,
                verbose = 1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=r"C:\Users\bozth\Documents\UCL\MRes_Project\QSM-ML\training\trained_models\best_var_NDI",
                save_weights_only=True,
                moitor='val_accuracy',
                mode='max',
                save_best_only=True
            )
        ]
    )


if __name__ == '__main__':
    main()
