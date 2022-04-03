import utils.def_volume as vol
import utils.kernels as kern

import numpy as np
import matplotlib.pyplot as plt

def main():
    radius = 12
    size = radius * 4
    threshold = 0.000000000001

    sphere = vol.sphere(radius, size)

    kernel = kern.dipole_kernel(size, threshold)

    phase_change = 8 * np.fft.ifftn(np.fft.fftn(sphere.image) * kernel.kernel)
    # noise = np.random.normal(0, strength, self.image.shape)
    # self.image = self.image + noise
    recon_sus_map = np.fft.ifftn(np.fft.fftn(phase_change / 8) * kernel.inv_kernel)

    line = np.real(phase_change)[radius * 2, :, radius * 2]

    img1 = np.abs(phase_change)[radius * 2,:,:]
    img2 = np.abs(phase_change)[:,radius * 2,:]
    img3 = np.abs(phase_change)[:,:,radius * 2]

    img4 = np.abs(recon_sus_map)[radius * 2,:,:]
    img5 = np.abs(recon_sus_map)[:,radius * 2,:]
    img6 = np.abs(recon_sus_map)[:,:,radius * 2]

    sus_plot = plt.imshow(sphere.image[radius * 2,:,:], cmap= 'gray', vmin = 0, vmax = 255)
    plt.colorbar()
    plt.show()

    # ft_sphere_plot = plt.imshow(np.abs(np.fft.fftn(sphere.image))[:,0,:], cmap= 'gray', vmin = 0, vmax = 255)
    # plt.colorbar()
    # plt.show()

    phase_plot1 = plt.imshow(img1, cmap= 'gray', vmin = 0, vmax = 255)
    plt.colorbar()
    plt.show()

    # phase_line = plt.plot(line)
    # plt.show()

    phase_plot2 = plt.imshow(img2, cmap= 'gray', vmin = 0, vmax = 255)
    plt.colorbar()
    plt.show()

    phase_plot3 = plt.imshow(img3, cmap= 'gray', vmin = 0, vmax = 255)
    plt.colorbar()
    plt.show()

    recon_sus_plot1 = plt.imshow(img4, cmap= 'gray', vmin = 0, vmax = 255)
    plt.colorbar()
    plt.show()

    recon_sus_plot2 = plt.imshow(img5, cmap= 'gray', vmin = 0, vmax = 255)
    plt.colorbar()
    plt.show()

    recon_sus_plot3 = plt.imshow(img6, cmap= 'gray', vmin = 0, vmax = 255)
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    main()