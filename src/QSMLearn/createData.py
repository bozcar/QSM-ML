from typing import List

import numpy as np

class TrainingData:
    def __init__(self, array: np.ndarray) -> None:
        self.array = array
        self.shape = array.shape

    @classmethod
    def shape_collection(cls, shape_list: List[np.ndarray]):
        return cls

    @staticmethod
    def sphere():
        
