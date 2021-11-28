from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

class phase_map:
    """
    # Phase Map
    A class for handling phase map data. 
    Can be initialised with a suceptibility map to create a simulation of the phase map which would be produced from the suceptibility distribution.
    """
    def __init__(self, file_path:Path):
        self.img = nib.load(file_path)
        self.img_data = self.img.get_fdata()

    def apply_mask(self, mask:np.ndarray):
        self.masked = self.img_data * mask

    def save(self):
        raise NotImplementedError

# pm_path = Path("Replication/Datasets/Frequency.nii")
# mask_path = Path("Replication/Datasets/MaskBrainExtracted.nii")

# img = nib.load(pm_path)
# mask = nib.load(mask_path)

# img_data = img.get_fdata()
# mask_data = img.get_fdata()

