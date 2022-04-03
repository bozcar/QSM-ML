from pathlib import Path

import nibabel as nib
import tensorflow as tf
import matplotlib.pyplot as plt

import models

def main():
    IMPATH = Path(r"C:\Users\bozth\Documents\UCL\MRes_Project\QSM-ML\Replication\Datasets\Frequency.nii")
    MPATH = Path(r"C:\Users\bozth\Documents\UCL\MRes_Project\QSM-ML\Replication\Datasets\MaskBrainExtracted.nii")

    img = nib.load(IMPATH)
    mask = nib.load(MPATH)

    img_tensor = tf.constant(img.get_fdata())
    mask_tensor = tf.constant(mask.get_fdata())

    NDIfixed = models.FixedStepNDI(100, name='fixed')

    susceptibility = NDIfixed(img_tensor, mask_tensor)

    plt.imshow(susceptibility[100, :, :], cmap='gray')
    plt.axes('off')
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    main()