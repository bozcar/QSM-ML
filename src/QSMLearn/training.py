def main():
    import matplotlib.pyplot as plt
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras

    from .models import FixedStepNDI, VariableStepNDI
    from .shapes import shapes

    PARAMS = {
        'nucleus' : 'H1',
        'B0' : 1.5,
        'TE' : 30e-3
    }

    REGION = (200, 250, 225)

    m = FixedStepNDI(100)
    sus = [shapes.sphere(REGION, (100, 125, 150), 50, 0.000006)]

    x_train = np.array([[s.get_phase(1).numpy(), np.ones(REGION)] for s in sus])
    y_train = np.array([s.get_tensor().numpy() for s in sus])

    m.compile(
        optimizer='adam',
        loss=keras.losses.MeanSquaredError()
    )

    history = m.fit(
        x = x_train,
        y = y_train,
        batch_size=1,
        epochs=10
    )

def training(model_choice: str, steps: int, *, x_train, y_train, **kwargs):
    from tensorflow import keras

    from .models import FixedStepNDI, VariableStepNDI

    MODELS = {
        'fs' : FixedStepNDI,
        'vs' : VariableStepNDI
    }

    m = MODELS[model_choice](steps, kwargs = kwargs)

    m.compile(
        optimizer='adam',
        loss=keras.losses.MeanSquaredError()
    )

    return m.fit(
        x = x_train,
        y = y_train, 
        batch_size = 1,
        epochs = 10
    )

def trainQSM():
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Train a network to find optimal QSM parameters.")

    parser.add_argument('model', choices=['fs', 'vs'])
    parser.add_argument('--steps', '-s', type=int, default=10)

    arguments = parser.parse_args()

    training(arguments.model, arguments.steps)

if __name__ == '__main__':
    trainQSM()