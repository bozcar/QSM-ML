from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import utils.kernels as kernels
import utils.phase_map as phase_map
import utils.image_handling as images

def main():
    pm_path = Path(r"C:\Users\bozth\Documents\UCL\MRes_Project\QSM-ML\data\Sim2Snr2\Frequency.nii.gz")
    mask_path = Path(r"C:\Users\bozth\Documents\UCL\MRes_Project\QSM-ML\data\Sim2Snr2\MaskBrainExtracted.nii.gz")

    pm = phase_map.phase_map(pm_path)
    mask = phase_map.phase_map(mask_path)

    kernel = kernels.dipole_kernel_nsq(np.shape(pm.img_data), 0.1)

    # images.show_image(kernel.kernel[0,:,:])
    # images.show_image(kernel.kernel[:,:,80])

    # images.show_image(kernel.inv_kernel[0,:,:])
    # images.show_image(kernel.inv_kernel[:,:,80])

    qsm = np.fft.ifftn(np.fft.fftn(pm.img_data/8) * kernel.inv_kernel)
    qsm_masked = qsm * mask.img_data

    img4 = plt.imshow(np.real(np.flip(qsm[82,:,:].T, axis=0)), cmap='gray')
    plt.axis('off')
    plt.show()

    # img5 = plt.imshow(np.real(qsm_masked[:,102,:]), cmap='gray')
    # plt.colorbar()
    # plt.show()

    # img6 = plt.imshow(np.real(qsm_masked[:,:,102]), cmap='gray')
    # plt.colorbar()
    # plt.show()

if __name__ == '__main__':
    main()