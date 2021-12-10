import numpy as np
import utils.kernels as kernels

def NDI_step(chi:np.ndarray, phase:np.ndarray, w:np.ndarray):
    D = kernels.dipole_kernel_nsq(np.shape(chi), 0.1)

    grad = 2 * np.fft.fftn(D.kernel * np.fft.ifftn(w * w * np.sin(np.fft.ifftn(D.kernel * np.fft.fftn(chi)) - phase)))
    chi_next = chi - grad

    # adjust weighting matrix so that corrupted pixels contribute less to the image reconstruction
    residual = w * np.abs(np.exp(1j*np.fft.ifftn(D.kernel*np.fft.fftn(chi_next))) - np.exp(1j*phase))
    standard_deviation = np.std(residual)

    w_next = w
    w_next[residual > 6 * standard_deviation] = w[residual > 6 * standard_deviation]/((residual[residual > 6 * standard_deviation]/standard_deviation)**2)
    
    return chi_next, w_next