if __name__ == '__main__':
    import numpy as np
    from tensorflow import keras

    from QSMLearn.models import VariableStepNDI

    x_train = np.load(r"C:\Users\bozth\Documents\UCL\MRes_Project\QSM-ML\data\x_train_1.npy")
    y_train = np.load(r"C:\Users\bozth\Documents\UCL\MRes_Project\QSM-ML\data\y_train_1.npy")

    m = VariableStepNDI(
        iters = 200
    )

    m.compile(
        optimizer = keras.optimizers.Adam(learning_rate = 0.5),
        loss = keras.losses.MeanSquaredError()
    )

    history = m.fit(
        x = x_train,
        y = y_train,
        validation_split = 0.2,
        batch_size = 3,
        epochs = 3000,
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