from enum import Enum
from dataclasses import dataclass
import numpy as np
import scipy.sparse as spsparse
import cupy as cp
import cupyx.scipy.sparse as cpsparse
import matplotlib.pyplot as plt
from typing import List
from typing import Tuple
from matplotlib.animation import FuncAnimation


class DimensionIndex(Enum):
    X = 0
    Y = 1
    Z = 2
    W = 3

class MeshGrid: 
    def __init__(self, gridDimensionalComponents : tuple[np.ndarray], pointCount : int, length : float): 
        self.pointCount = pointCount
        self.length = length
        self.gridDimensionalComponents : tuple[np.ndarray] = gridDimensionalComponents 
        self.dimensions = len(self.gridDimensionalComponents)
        for dimension_ in list(DimensionIndex.__members__): 
            dimension = getattr(DimensionIndex, dimension_)
            if self.dimensions > dimension.value: 
                setattr(self, dimension.name.lower(), self.gridDimensionalComponents[dimension.value])
        self.asArray = None
    def toArray(self) -> np.array: 
        self.asArray = np.column_stack(np.array([
                component.ravel() \
                for component in self.gridDimensionalComponents
            ])).ravel()
        return self.asArray

def makeLinspaceGrid(pointCount : int, length : float, dimensions : int, halfSpaced = False, componentType : type = float) -> MeshGrid: 
    if halfSpaced == True: 
        spaces : tuple[np.array] = tuple((np.linspace(-length / 2, length  / 2, pointCount, dtype = componentType) for ii in range(dimensions)))
    else: 
        spaces : tuple[np.array] = tuple((np.linspace(0, length, pointCount, dtype = componentType) for ii in range(dimensions)))
    return MeshGrid(np.meshgrid(*spaces), pointCount, length)

def applyEdge(grid : MeshGrid, value : float = 0.0) -> MeshGrid: 
    grid[0, :] = value
    grid[-1, :] = value
    grid[:, 0] = value
    grid[:, -1] = value
    return grid

class SimulationProfile: 
    def __init__(
                self, 
                grid : MeshGrid, 
                initialWaveFunctionGenerator, 
                potentialGenerator, 
                timeStep, 
                spaceStep, 
                gpuAccelerated = False
            ): 
        self.grid = grid
        self.dimensions = self.grid.dimensions
        self.initialWaveFunctionGenerator = initialWaveFunctionGenerator
        self.potentialGenerator = potentialGenerator
        self.timeStep = timeStep
        self.spaceStep = spaceStep
        self.sparse = spsparse if gpuAccelerated == False else cpsparse
        self.math = np if gpuAccelerated == False else cp

class SimulationResults: 
    def __init__(self, waveFunction : MeshGrid): 
        self.waveFunctions = waveFunction if type(waveFunction) is list else [waveFunction]

def placeDiagonals(
                center : np.array, 
                inner : np.array, 
                outer : np.array, 
                rowLength : int, 
                math, 
                sparse
            ): 
        return sparse.spdiags(
            math.array([
                    outer, 
                    inner, 
                    center, 
                    inner, 
                    outer
                ]),
            math.array([-rowLength, -1, 0, 1, rowLength]), 
            rowLength, 
            rowLength
        )

def toDiagonal(vector : np.array, math, sparse): 
    return sparse.spdiags( 
            math.array([vector]), 
            math.array([0]), 
            len(vector), 
            len(vector)
        )

def createStepMatrix(simulator, currentPotential, scalar, innerDiagonal, outerDiagonal):
    math = simulator.math
    sparse = simulator.sparse
    unknownFactorCount = simulator.unknownFactorCount
    centerDiagonal = (1j * simulator.timeStep / 2) * currentPotential[1:-1, 1:-1].ravel()
    centerDiagonal = 1 + math.sum(scalar * simulator.stepConstants * 2.0) + (scalar * centerDiagonal)
    diagonalLength = len(centerDiagonal) - 2
    #print("V:", currentPotential[1:-1, 1:-1].ravel())
    #print("C:", centerDiagonal)
    stepMatrix = placeDiagonals(
                centerDiagonal, 
                innerDiagonal, 
                outerDiagonal, 
                unknownFactorCount, 
                math, 
                sparse
            )
    return stepMatrix

def createCurrentStepMatrix(simulator, currentPotential): 
    return createStepMatrix(
            simulator, 
            currentPotential, 
            -1, 
            simulator.knownInnerDiagonal, 
            simulator.knownOuterDiagonal
        )

def createNextStepMatrix(simulator, nextPotential): 
    return createStepMatrix(
            simulator, 
            nextPotential, 
            1, 
            simulator.unknownInnerDiagonal, 
            simulator.unknownOuterDiagonal
        )

class Simulator: 
    def __init__(self, profile : SimulationProfile): 
        self.profile = profile
        self.grid = self.profile.grid
        self.dimensions = self.profile.dimensions
        self.timeStep = self.profile.timeStep
        self.spaceStep = self.profile.spaceStep
        self.math = self.profile.math
        self.sparse = self.profile.sparse
        self.stepConstants = np.ones(self.dimensions) * (-self.timeStep / (2j * self.profile.spaceStep ** 2))
        self.waveFunctions = [applyEdge(self.profile.initialWaveFunctionGenerator(self.grid))]
        self.potentials = [
                self.profile.potentialGenerator(self.grid, 0.0), 
                self.profile.potentialGenerator(self.grid, self.timeStep)
            ]
        self.unknownFactorCount = (self.grid.pointCount - 2) ** self.dimensions
        self.diagonalLength = self.unknownFactorCount #(len(self.potentials[-1]) - 2) * (len(self.potentials[-1][0]) - 2)
        self.knownInnerDiagonal = self.math.ones(self.diagonalLength) * self.stepConstants[0]
        self.knownOuterDiagonal = self.math.ones(self.diagonalLength) * self.stepConstants[1]
        self.unknownInnerDiagonal = -1 * self.knownInnerDiagonal # These two just to save a bit of compute
        self.unknownOuterDiagonal = -1 * self.knownOuterDiagonal # time multiplying these large arrays by -1
        #stepMatracies = computeStepMatricies(self)

    def compute(self, time, maxTime, unknownStepMatrix, knownStepMatrix): 
        math = self.math
        sparse = self.sparse
        waveFunctionVector = self.waveFunctions[-1][1:-1, 1:-1].reshape((self.diagonalLength, 1))
        #waveFunctionVector = self.waveFunctions[-1]
        independantTerms = knownStepMatrix * waveFunctionVector #math.matmul(knownStepMatrix, waveFunctionVector)
        nextWaveFunction = sparse.linalg.spsolve(sparse.csc_matrix(unknownStepMatrix), independantTerms).reshape(
                tuple([self.grid.pointCount - 2] * self.dimensions), 
            )
        self.waveFunctions.append(np.pad(nextWaveFunction, 1))
        return self.waveFunctions[-1], self.potentials[-1]

    def simulate(self, maxTime): 
        math = self.math
        timeStep = self.timeStep
        timePoints = round(maxTime / timeStep)
        time = timeStep
        for ii in range(1, timePoints): 
            self.potentials.append(
                    self.profile.potentialGenerator(self.grid, (ii + 1) * self.timeStep)
                )
            knownStepMatrix = createCurrentStepMatrix(self, self.potentials[-2])
            unknownStepMatrix = createNextStepMatrix(self, self.potentials[-1])
            self.compute(time, maxTime, unknownStepMatrix, knownStepMatrix)
            time = (ii + 1) * timeStep

    def processProbabilities(self): 
        math = self.math
        self.probabilities = np.array(list(map(
                lambda waveFunction :  math.sqrt(math.real(waveFunction) ** 2 + math.imag(waveFunction) ** 2).astype(math.float64), 
                self.waveFunctions
            )))
        self.probabilityDecibles = 0
        #self.probabilityDecibles = np.array(list(map(
        #        lambda probabilities : -10 * math.log10(probabilities), 
        #        self.probabilities
        #    )))
        return self.probabilities, self.probabilityDecibles

def makeWavePacket(grid, startX, startY, sigma=0.5, k = 15 * np.pi): 
    return np.exp((-1 / (2 * sigma ** 2)) * ((grid.x - startX) ** 2 + (grid.y - startY) ** 2)) \
            * np.exp(-1j * k * (grid.x - startX))
def animateImages(pointCount : int, images : List[np.array]): 
    animationFigure = plt.figure()
    animationAxis = animationFigure.add_subplot(xlim=(0, pointCount), ylim=(0, pointCount))
    animationFrame = animationAxis.imshow(images[0])
    def animateFrame(frameIndex): 
        animationFrame.set_data(images[frameIndex])
        animationFrame.set_zorder(1)
        return animationFrame,
    animation = FuncAnimation(animationFigure, animateFrame, interval=1, frames=np.arange(0, len(images), 2), repeat = True, blit = 0)
    return animation

if __name__ == "__main__": 
    pointCount : int = 50
    profile = SimulationProfile(
            makeLinspaceGrid(pointCount, 1, 2), 
            makeWavePacket, 
            lambda position, time : np.sqrt(position.x ** 2 + position.y ** 2), 
            .01, 
            .01
        )
    simulator = Simulator(profile)
    simulator.simulate(1)
    probabilities, probabilityDecibles = simulator.processProbabilities()
    print(simulator.waveFunctions[-1])
    #plt.imshow(simulator.waveFunctions[-1])
    plt.imshow(simulator.probabilities[-1])

