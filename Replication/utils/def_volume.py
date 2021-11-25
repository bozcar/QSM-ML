import numpy as np

class sphere:
    def __init__(self, radius, size, noise = False, noise_strength = 0):
        
        self.image = np.zeros((size, size, size), dtype=np.int8)
        self.vertices = np.zeros((size + 1, size + 1, size + 1), dtype=np.int8)
        offset = size/2
        
        for i in range(size + 1):
            for j in range(size + 1):
                for k in range(size + 1):
                    if (i - offset)**2 + (j - offset)**2 + (k - offset)**2 < radius**2:
                        self.vertices[i,j,k] = 1

        for i in range(size):
            for j in range(size):
                for k in range(size):
                    self.image[i,j,k] = np.sum(self.vertices[i:i+2, j:j+2, k:k+2]*12)

        if noise == True:
            self.add_noise(noise_strength)

    def add_noise(self, strength):
        noise = np.random.normal(0, strength, self.image.shape)
        self.image = self.image + noise
