"""

"""


from pathlib import Path


def get_data(dirname: str, shape):
    from QSMLearn import tfrecords

    datadir = Path(dirname)
    datapaths = datadir.glob('*.tfrecords')
    datafilenames = [str(path) for path in datapaths]
    
    return tfrecords.read_and_decode(datafilenames, shape)


def main():
    import datetime

    from tensorflow import keras

    from QSMLearn import models

    # find all tfrecords files in `DATALIB`
    BASE_DIR = r"D:\48data_normal4"
    SHAPE = [48, 48, 48]

    training_dataset = get_data(BASE_DIR + r'\train', SHAPE)
    training_dataset = training_dataset.shuffle(500)
    training_dataset = training_dataset.batch(12, drop_remainder=False)

    validation_dataset = get_data(BASE_DIR + r'\test', SHAPE)
    validation_dataset = validation_dataset.batch(12, drop_remainder=False)

    log_dir = Path(r"C:\Users\bozth\Documents\UCL\MRes_Project\QSM-ML\training\logs2")
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    run_path = log_dir / timestamp

    # NDI model with 200 trainable step sizes
    m = models.VariableStepNDI(
        iters=200,
        verbose = False,
        mode = 'r'
    )

    m.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-2),
        loss=keras.losses.MeanSquaredError(),
    )
    history = m.fit(
        x = training_dataset,
        validation_data=validation_dataset,
        epochs = 100,
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience = 5,
                monitor = 'val_loss',
                verbose = 1
            ),
            keras.callbacks.ReduceLROnPlateau(
                factor = 0.5, 
                patience = 3,
                monitor = 'val_loss',
                min_lr = 0,
                verbose = 1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=str(run_path / 'checkpoint'),
                save_weights_only=True,
                moitor='val_accuracy',
                mode='max',
            ),
            keras.callbacks.TensorBoard(
                log_dir=run_path,
                histogram_freq=1
            )
        ]
    )


if __name__ == '__main__':
    main()
