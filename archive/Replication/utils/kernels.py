import numpy as np

class dipole_kernel:

    def __init__(self, size, threshold):
        half = size/2
        ONE_THIRD = 1/3

        self.kernel = np.fft.fftshift(np.array([[[ONE_THIRD - (z**2/((x-half)**2 + (y-half)**2 + (z-half)**2)) if x or y or z else 0 for x in range(size)] for y in range(size)] for z in range(size)]))
        self.create_inv_kernel(size, threshold)

    def create_inv_kernel(self, size, threshold):
        self.inv_kernel = np.zeros((size, size, size))

        for i in range(size):
            for j in range(size):
                for k in range(size):
                    if 0 <= self.kernel[i,j,k] < threshold:
                        self.inv_kernel[i,j,k] = 1 / threshold

                    elif 0 > self.kernel[i,j,k] > -1 * threshold:
                        self.inv_kernel[i,j,k] = -1 / threshold

                    else:
                        self.inv_kernel[i,j,k] = 1 / self.kernel[i,j,k]
    
class dipole_kernel_nsq:
    """
    # Dipole Kernel
    Handles the creation of a dipole kernel in its k-space formulation. 
    Initialises with both the kernel and its inverse which is truncated in positions where the absulute value of the kernel is less than the threshold value.
    The value of the inverse kernel in the truncated positions are 1/threshold (or -1/threshold where the truncated value was negative).  
    """
    def __init__(self, size:list, threshold:float):
        ONE_THIRD = 1/3

        x = np.arange(-1, 1, 2 / size[0])
        y = np.arange(-1, 1, 2 / size[1])
        z = np.arange(-1, 1, 2 / size[2])
        
        vx, vy, vz = np.meshgrid(x, y, z, sparse=False, indexing='ij')

        denom = vx**2 + vy**2 + vz**2
        z_squared = vz**2

        self.kernel = np.zeros(size)
        self.kernel[denom != 0] = ONE_THIRD - (z_squared[denom != 0]/denom[denom != 0])
        self.kernel = np.fft.fftshift(self.kernel)
        
        self.create_inv_kernel(threshold)

    def create_inv_kernel(self, threshold:float):
        self.inv_kernel = np.zeros(np.shape(self.kernel))

        self.inv_kernel[np.logical_and(0 <= self.kernel, self.kernel< threshold)] = 1 / threshold
        self.inv_kernel[np.logical_and(0 > self.kernel, self.kernel > -1 * threshold)] = -1 / threshold
        self.inv_kernel[np.logical_or(threshold <= self.kernel, -1 * threshold >= self.kernel)] = 1 / self.kernel[np.logical_or(threshold <= self.kernel, -1 * threshold >= self.kernel)]