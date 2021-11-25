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
    def __init__(self, SM, mode:str = 's'):
        """
        Can be initialised in various modes:
        - s - Simulated from a suceptibility map
        - d - Directly loaded from a file
        """
        if mode.lower() == 's':
            #TODO: implement simulation init
            raise NotImplementedError
        elif mode.lower() == 'd':
            #TODO: implement direct init
            raise NotImplementedError
        else:
            raise ValueError(f"Mode {mode} is not a recognised mode.")

    def save(self):
        raise NotImplementedError

pm_path = Path("Datasets/Frequency.nii")
mask_path = Path("Datasets/MaskBrainExtracted.nii")

img = nib.load(pm_path)
mask = nib.load(mask_path)

img_data = img.get_fdata()
mask_data = img.get_fdata()

