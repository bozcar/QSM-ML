from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import utils.kernels as kernels
import utils.phase_map as phase_map
import utils.image_handling as images

def main():
    pm_path = Path("Replication/Datasets/Frequency.nii")
    mask_path = Path("Replication/Datasets/MaskBrainExtracted.nii")

    pm = phase_map.phase_map(pm_path)
    mask = phase_map.phase_map(mask_path)

    kernel = kernels.dipole_kernel_nsq(np.shape(pm.img_data), 0.15)

    images.show_image(kernel.kernel[0,:,:])

    qsm = np.fft.ifftn(np.fft.fftn(pm.img_data/8) * kernel.inv_kernel)
    qsm_masked = qsm * mask.img_data

    img4 = plt.imshow(np.real(qsm_masked[82,:,:]), cmap='gray')
    plt.colorbar()
    plt.show()

    img5 = plt.imshow(np.real(qsm_masked[:,102,:]), cmap='gray')
    plt.colorbar()
    plt.show()

    img6 = plt.imshow(np.real(qsm_masked[:,:,102]), cmap='gray')
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    main()