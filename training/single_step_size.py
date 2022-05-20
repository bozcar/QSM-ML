if __name__ == '__main__':
    import numpy as np
    from tensorflow import keras

    from QSMLearn.models import FixedStepNDI

    x = np.load(r"C:\Users\bozth\Documents\UCL\MRes_Project\QSM-ML\data\x_train_1.npy")

    SHAPE = x.shape[1:]

    x_train = np.array([[item, np.ones(SHAPE)] for item in x])
    y_train = np.load(r"C:\Users\bozth\Documents\UCL\MRes_Project\QSM-ML\data\y_train_1.npy")

    m = FixedStepNDI(
        iters = 20,
        init_step = 1.
    )

    m.compile(
        optimizer = keras.optimizers.Adam(learning_rate = 1.),
        loss = keras.losses.MeanSquaredError()
    )

    history = m.fit(
        x = x_train,
        y = y_train,
        validation_split = 0.2,
        batch_size = 2,
        epochs = 300,
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience = 10,
                monitor = 'val_loss',
                verbose = 1
            ),
            keras.callbacks.ReduceLROnPlateau(
                factor = 0.5, 
                patience = 3,
                min_lr = 0,
                verbose = 1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=r"C:\Users\bozth\Documents\UCL\MRes_Project\QSM-ML\training\trained_models\best_model",
                save_weights_only=True,
                moitor='val_accuracy',
                mode='max',
                save_best_only=True
            )
        ]
    )