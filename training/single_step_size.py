if __name__ == '__main__':
    import numpy as np
    from tensorflow import keras

    from QSMLearn.models import FixedStepNDI
    from QSMLearn.shapes import shapes

    REGION = (128, 128, 128)
    CENTER = (63.5, 63.5, 63.5)
    RADII = [32, 30, 28, 26, 24, 22, 20, 18, 16, 14]

    sus = [shapes.sphere(REGION, CENTER, r, 1) for r in RADII]

    x_train = np.array([[s.get_phase(1).numpy(), np.ones(REGION)] for s in sus])
    y_train = np.array([s.get_tensor().numpy() for s in sus])

    m = FixedStepNDI(
        iters = 350,
        init_step = 1.
    )

    m.compile(
        optimizer = keras.optimizers.Adam(learning_rate = 1),
        loss = keras.losses.MeanSquaredError()
    )

    history = m.fit(
        x = x_train,
        y = y_train,
        batch_size = 10,
        epochs = 300
    )