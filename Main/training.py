def main():
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from tensorflow import keras

    from models import FixedStepNDI, VariableStepNDI
    from shapes import shapes

    m = FixedStepNDI(250)
    sus = shapes.sphere((25, 25, 25), (11.5, 11.5, 11.5), 6, 0.000001)

    m.compile(
        optimizer='adam',
        loss=keras.losses.MeanSquaredError()
    )

    history = m.fit(
        
    )

if __name__ == '__main__':
    main()