import numpy as np

class dipole_kernel:

    def __init__(self, size, threshold):
        half = size/2
        ONE_THIRD = 1/3

        self.kernel = np.fftshift(np.array([[[ONE_THIRD - (z**2/((x-half)**2 + (y-half)**2 + (z-half)**2)) if x or y or z else 0 for x in range(size)] for y in range(size)] for z in range(size)]))
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

    def __init__(self, size:list):
        ONE_THIRD = 1/3
