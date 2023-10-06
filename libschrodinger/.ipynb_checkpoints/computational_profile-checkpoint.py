import numpy as np
import cupy as cp
import scipy.sparse as spsparse
import cupyx.scipy.sparse as cpsparse
import scipy.sparse.linalg as spla
import cupyx.scipy.sparse.linalg as cpla

class ComputationalProfile: 
    def __init__(self, gpuAccelerated : bool):
        self.sparse = spsparse if gpuAccelerated == False else cpsparse
        self.math = np if gpuAccelerated == False else cp
        self.linalg = spla if gpuAccelerated == False else cpla

