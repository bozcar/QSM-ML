def main():
    from QSMLearn import shapes
    import numpy as np

    s = shapes.shapes.sphere(
        shape=(150, 150, 150),
        centre=(75, 75, 75),
        radius=25
    )

    sus = s.get_tensor().numpy()
    phase = s.get_phase(1).numpy()

    suses = [sus[i:i+64, i:i+64, i:i+64] for i in range(45, 55)]
    phases = [phase[i:i+64, i:i+64, i:i+64] for i in range(45, 55)]

    np.save(r"C:\Users\bozth\Documents\UCL\MRes_Project\QSM-ML\data\x_train_1", suses)
    np.save(r"C:\Users\bozth\Documents\UCL\MRes_Project\QSM-ML\data\y_train_1", phases)


if __name__ == '__main__':
    main()
