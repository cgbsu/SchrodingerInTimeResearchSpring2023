from enum import Enum
import numpy as np
from typing import Tuple, List, Dict

class DimensionIndex(Enum):
    X = 0
    Y = 1
    Z = 2
    W = 3

class MeshGrid: 
    def __init__(self, gridDimensionalComponents : Tuple[np.ndarray], pointCount : int, length : float, math = np): 
        self.math = math
        self.pointCount = pointCount
        self.length = length
        self.gridDimensionalComponents : Tuple[np.ndarray] = gridDimensionalComponents 
        self.dimensions = len(self.gridDimensionalComponents)
        for dimension_ in list(DimensionIndex.__members__): 
            dimension = getattr(DimensionIndex, dimension_)
            if self.dimensions > dimension.value: 
                setattr(self, dimension.name.lower(), self.gridDimensionalComponents[dimension.value])
        self.asArray = None
    def toArray(self) -> np.array: 
        math = self.math
        self.asArray = math.column_stack(math.array([
                component.ravel() \
                for component in self.gridDimensionalComponents
            ])).ravel()
        return self.asArray

def makeLinspaceGrid(
            pointCount : int, 
            length : float, 
            dimensions : int, 
            halfSpaced = False, 
            componentType : type = float, 
            math = np
        ) -> MeshGrid: 
    if halfSpaced == True: 
        spaces : Tuple[np.array] = tuple(
                (math.linspace(-length / 2, length  / 2, pointCount, dtype = componentType) \
                        for ii in range(dimensions))
            )
    else: 
        spaces : Tuple[np.array] = tuple(
                (math.linspace(0, length, pointCount, dtype = componentType) \
                        for ii in range(dimensions))
            )
    return MeshGrid(math.meshgrid(*spaces), pointCount, length, math)

def applyEdge(grid : MeshGrid, value : float = 0.0) -> MeshGrid: 
    grid[0, :] = value
    grid[-1, :] = value
    grid[:, 0] = value
    grid[:, -1] = value
    return grid

