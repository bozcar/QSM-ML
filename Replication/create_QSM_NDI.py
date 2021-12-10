import numpy as np
from pathlib import Path

import utils.phase_map as phase_map
import utils.NDI as NDI
import utils.image_handling as image_handling

def main():
    pm_path = Path("Replication/Datasets/Frequency.nii")
    mask_path = Path("Replication/Datasets/MaskBrainExtracted.nii")

    pm = phase_map.phase_map(pm_path)
    mask = phase_map.phase_map(mask_path)

    qsm = np.zeros(np.shape(pm.img_data))
    w = np.ones(np.shape(qsm)) # weighting matrix
    iterations = 100

    for _ in range(iterations):
        qsm, w = NDI.NDI_step(qsm, pm.img_data, w)

    image_handling.show_image(np.real(qsm[:,:,105]))

if __name__ == '__main__':
    main()