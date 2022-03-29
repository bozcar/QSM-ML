def main():
    from random import random, uniform

    import matplotlib.pyplot as plt
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras

    from models import FixedStepNDI, VariableStepNDI
    from shapes import shapes

    PARAMS = {
        'nucleus' : 'H1',
        'B0' : 1.5,
        'TE' : 30e-3
    }

    REGION = (200, 250, 300)

    m = FixedStepNDI(1)
    sus = [shapes.sphere(REGION, (100, 125, 150), 50, 0.000006)]

    x_train = np.array([[s.get_phase(10).numpy(), s.get_mask().numpy()] for s in sus])
    y_train = np.array([s.get_tensor().numpy() for s in sus])

    plt.imshow(tf.math.real(x_train)[0, 0, 100, :, :], cmap='gray')
    plt.colorbar()
    plt.show()

    plt.imshow(tf.math.real(y_train)[0, 100, :, :], cmap='gray')
    plt.colorbar()
    plt.show()

    plt.imshow(tf.math.real(m(x_train))[0, 100, :, :], cmap='gray', vmin=0, vmax=3e-6)
    plt.colorbar()
    plt.show()

    # m.compile(
    #     optimizer='adam',
    #     loss=keras.losses.MeanSquaredError()
    # )

    # history = m.fit(
    #     x = x_train,
    #     y = y_train,
    #     batch_size=1,
    #     epochs=10
    # )

if __name__ == '__main__':
    main()