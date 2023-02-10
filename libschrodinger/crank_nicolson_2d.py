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
import numba
from numba import njit
from numba import jit
import scipy.sparse.linalg as spla
import cupyx.scipy.sparse.linalg as cpla


class DimensionIndex(Enum):
    X = 0
    Y = 1
    Z = 2
    W = 3

class MeshGrid: 
    def __init__(self, gridDimensionalComponents : tuple[np.ndarray], pointCount : int, length : float, math = np): 
        self.math = math
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
        math = self.math
        self.asArray = math.column_stack(math.array([
                component.ravel() \
                for component in self.gridDimensionalComponents
            ])).ravel()
        return self.asArray

def makeLinspaceGrid(pointCount : int, length : float, dimensions : int, halfSpaced = False, componentType : type = float, math = np) -> MeshGrid: 
    if halfSpaced == True: 
        spaces : tuple[np.array] = tuple((math.linspace(-length / 2, length  / 2, pointCount, dtype = componentType) for ii in range(dimensions)))
    else: 
        spaces : tuple[np.array] = tuple((math.linspace(0, length, pointCount, dtype = componentType) for ii in range(dimensions)))
    return MeshGrid(math.meshgrid(*spaces), pointCount, length, math)

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
                gpuAccelerated = False, 
                edgeBound = False
            ): 
        self.grid = grid
        self.dimensions = self.grid.dimensions
        self.initialWaveFunctionGenerator = initialWaveFunctionGenerator
        self.potentialGenerator = potentialGenerator
        self.timeStep = timeStep
        self.spaceStep = spaceStep
        self.sparse = spsparse if gpuAccelerated == False else cpsparse
        self.math = np if gpuAccelerated == False else cp
        self.linalg = spla if gpuAccelerated == False else cpla
        self.edgeBound = edgeBound

class SimulationResults: 
    def __init__(self, waveFunction : MeshGrid): 
        self.waveFunctions = waveFunction if type(waveFunction) is list else [waveFunction]

def placeDiagonals(
                center : np.array, 
                innerShifted : np.array, 
                inner : np.array, 
                outer : np.array, 
                rowLength : int, 
                extent : int, 
                math, 
                sparse
            ): 
        return sparse.spdiags(
                math.array([ 
                        outer, 
                        inner, 
                        center, 
                        innerShifted, 
                        outer
                    ]),
                math.array([-extent, -1, 0, 1, extent]), 
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

def createDenseStepMatrixImplementationBounded(
            unknownFactorCount, 
            timeStep, 
            stepConstants, 
            currentPotential, 
            stepMatrix, 
            scalar, 
            innerDiagonalShifted, 
            innerDiagonal, 
            outerDiagonal
        ):
    extent = len(currentPotential)
    rx = scalar * stepConstants[0]
    ry = scalar * stepConstants[1]
    for kk in range(unknownFactorCount): 
        ii = 1 + kk // (extent - 2)
        jj = 1 + kk % (extent - 2)
        stepMatrix[kk, kk] = 1 + 2 * rx + 2 * ry \
                + ((scalar * 1j * timeStep / 2) * currentPotential[ii, jj])
        if ii != 1: 
            stepMatrix[kk, (ii - 2) * (extent - 2) + jj - 1] = outerDiagonal[kk]#ry
        if ii != (extent - 2): 
            stepMatrix[kk, ii * (extent - 2) + jj - 1] = outerDiagonal[kk]#ry
        if jj != 1: 
            stepMatrix[kk, kk - 1] = innerDiagonalShifted[kk]#rx
        if jj != (extent - 2): 
            stepMatrix[kk, kk + 1] = innerDiagonal[kk]#rx
    return stepMatrix

def createDenseStepMatrixImplementation(
            unknownFactorCount, 
            timeStep, 
            stepConstants, 
            currentPotential, 
            stepMatrix, 
            scalar, 
            innerDiagonalShifted, 
            innerDiagonal, 
            outerDiagonal
        ):
    extent = len(currentPotential)
    rx = scalar * stepConstants[0]
    ry = scalar * stepConstants[1]
    for kk in range(unknownFactorCount): 
        ii = 1 + kk // (extent - 2)
        jj = 1 + kk % (extent - 2)
        stepMatrix[kk, kk] = 1 + 2 * rx + 2 * ry \
                + ((scalar * 1j * timeStep / 2) * currentPotential[ii, jj])
        if ii != 1: 
            stepMatrix[kk, (ii - 2) * (extent - 2) + jj - 1] = outerDiagonal[kk]#ry
        if ii != (extent - 2): 
            stepMatrix[kk, ii * (extent - 2) + jj - 1] = outerDiagonal[kk]#ry
        if kk > 0: 
            stepMatrix[kk, kk - 1] = innerDiagonalShifted[kk]#rx
        if kk < unknownFactorCount - 1: 
            stepMatrix[kk, kk + 1] = innerDiagonal[kk]#rx
    return stepMatrix

def createDenseStepMatrixBounded(
            simulator, 
            currentPotential, 
            scalar, 
            shiftedInnerDiagonal, 
            innerDiagonal, 
            outerDiagonal
        ):
    unknownFactorCount = simulator.unknownFactorCount
    stepMatrix = simulator.math.zeros((unknownFactorCount, unknownFactorCount), complex)
    return createDenseStepMatrixImplementationBounded(
            unknownFactorCount, 
            simulator.timeStep, 
            simulator.stepConstants, 
            currentPotential, 
            stepMatrix, 
            scalar, 
            shiftedInnerDiagonal, 
            innerDiagonal, 
            outerDiagonal
        )

def createDenseStepMatrix(
            simulator, 
            currentPotential, 
            scalar, 
            shiftedInnerDiagonal, 
            innerDiagonal, 
            outerDiagonal
        ):
    unknownFactorCount = simulator.unknownFactorCount
    stepMatrix = simulator.math.zeros((unknownFactorCount, unknownFactorCount), complex)
    return createDenseStepMatrixImplementation(
            unknownFactorCount, 
            simulator.timeStep, 
            simulator.stepConstants, 
            currentPotential, 
            stepMatrix, 
            scalar, 
            shiftedInnerDiagonal, 
            innerDiagonal, 
            outerDiagonal
        )

def createStepMatrix(
            simulator, 
            currentPotential, 
            scalar, 
            shiftedInnerDiagonal, 
            innerDiagonal, 
            outerDiagonal
        ):
    math = simulator.math
    sparse = simulator.sparse
    unknownFactorCount = simulator.unknownFactorCount
    centerDiagonal = (scalar * 1j * simulator.timeStep / 2) * currentPotential[1:-1, 1:-1].ravel()
    centerDiagonal = 1 + math.sum(scalar * simulator.stepConstants * 2.0) + centerDiagonal
    stepMatrix = placeDiagonals(
            centerDiagonal, 
            shiftedInnerDiagonal, 
            innerDiagonal, 
            outerDiagonal, 
            unknownFactorCount, 
            len(currentPotential) - 2, 
            math, 
            sparse
        )
    return stepMatrix

def createCurrentStepMatrix(simulator, currentPotential): 
    return createStepMatrix(
            simulator, 
            currentPotential, 
            1, 
            simulator.knownInnerDiagonalShifted, 
            simulator.knownInnerDiagonal, 
            simulator.knownOuterDiagonal
        )

def createNextStepMatrix(simulator, nextPotential): 
    return createStepMatrix(
            simulator, 
            nextPotential, 
            -1, 
            simulator.unknownInnerDiagonalShifted, 
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
        self.linalg = self.profile.linalg
        self.stepConstants = self.math.ones(self.dimensions) \
                * (-self.timeStep / (2j * self.profile.spaceStep ** 2))
        self.waveFunctions = [
                self.profile.initialWaveFunctionGenerator(self.grid)
            ]
        self.potentials = [
                self.profile.potentialGenerator(self.grid, 0.0), 
                self.profile.potentialGenerator(self.grid, self.timeStep)
            ]
        self.unknownFactorCount = (self.grid.pointCount - 2) ** self.dimensions
        self.diagonalLength = self.unknownFactorCount
        self.knownInnerDiagonal = self.math.ones(self.diagonalLength) \
                * self.stepConstants[0]
        if self.profile.edgeBound == True: 
            extent = int(self.math.sqrt(self.diagonalLength))
            self.knownInnerDiagonal.reshape((extent, extent)).T[-1] = 0
            self.knownInnerDiagonal = self.knownInnerDiagonal.reshape(self.diagonalLength)
        self.knownInnerDiagonalShifted = self.math.roll(self.knownInnerDiagonal, 1)
        self.knownOuterDiagonal = self.math.ones(self.diagonalLength) \
                * self.stepConstants[1]
        self.unknownInnerDiagonal = -1 * self.knownInnerDiagonal # These two just to save a bit of compute
        self.unknownOuterDiagonal = -1 * self.knownOuterDiagonal # time multiplying these large arrays by -1
        self.unknownInnerDiagonalShifted = -1 * self.knownInnerDiagonalShifted

    def compute(self, time, unknownStepMatrix, knownStepMatrix): 
        math = self.math
        sparse = self.sparse
        waveFunctionVector = self.waveFunctions[-1][1:-1, 1:-1].reshape(
                (self.diagonalLength, 1)
            )
        independantTerms = knownStepMatrix @ waveFunctionVector 
        #independantTerms = math.matmul(knownStepMatrix, waveFunctionVector)
        nextWaveFunction = self.linalg.spsolve(
                    sparse.csr_matrix(unknownStepMatrix), 
                    independantTerms
            ).reshape(tuple([self.grid.pointCount - 2] * self.dimensions),)
        self.waveFunctions.append(math.pad(nextWaveFunction, 1))
        return self.waveFunctions[-1], self.potentials[-1]

    def simulate(self, timePoints : int, printProgress : bool = False): 
        math = self.math
        timeStep = self.timeStep
        #timePoints = round(maxTime / timeStep)
        time = timeStep
        for ii in range(1, timePoints): 
            self.potentials.append(
                    self.profile.potentialGenerator(self.grid, (ii + 1) * self.timeStep)
                )
            knownStepMatrix = createCurrentStepMatrix(self, self.potentials[-2])
            unknownStepMatrix = createNextStepMatrix(self, self.potentials[-1])
            self.compute(time, unknownStepMatrix, knownStepMatrix)
            time = (ii + 1) * timeStep
            if printProgress == True: 
                print("(" + str(ii) + "/" + str(timePoints) + ")")

    def processProbabilities(self): 
        math = self.math
        self.probabilities = math.array(list(map(
                lambda waveFunction : math.sqrt( \
                        math.real(waveFunction) ** 2 + math.imag(waveFunction) ** 2
                    ).astype(math.float64), 
                self.waveFunctions
            )))
        self.probabilityDecibles = 0
        #self.probabilityDecibles = np.array(list(map(
        #        lambda probabilities : -10 * math.log10(probabilities), 
        #        self.probabilities
        #    )))
        return self.probabilities, self.probabilityDecibles

def makeWavePacket(grid, startX, startY, spatialStep, sigma=0.5, k = 15 * np.pi, math = np): 
    unnormalized = math.exp((-1 / (2 * sigma ** 2)) * ((grid.x - startX) ** 2 + (grid.y - startY) ** 2)) \
            * math.exp(-1j * k * (grid.x - startX))
    totalProbability = math.sum(math.sqrt(math.real(unnormalized) ** 2 + math.imag(unnormalized) ** 2))
    return unnormalized / totalProbability

def animateImages(pointCount : int, images : List[np.array], interval = 1): 
    animationFigure = plt.figure()
    animationAxis = animationFigure.add_subplot(xlim=(0, pointCount), ylim=(0, pointCount))
    animationFrame = animationAxis.imshow(images[0])
    def animateFrame(frameIndex): 
        animationFrame.set_data(images[frameIndex])
        animationFrame.set_zorder(1)
        return animationFrame,
    animation = FuncAnimation(
            animationFigure, 
            animateFrame, 
            interval=interval, 
            frames=np.arange(0, len(images), 2), 
            repeat = True, 
            blit = 0
        )
    return animation
    
def totalProbabilityInRegion(
            probabilityFrames : np.array, 
            pointCount : int, 
            spatialStep : float, 
            x : float, 
            y : float, 
            width : float, 
            height : float, 
            math = np
        ) -> np.array: 
    extent = pointCount
    normalizationValues = np.array(list(map(lambda frame : 
            (spatialStep ** 2) * math.sum(math.sum(frame)), 
            probabilityFrames
        )))
    cutFrames = np.array(list(map(lambda frame : 
            frame[
                    int(y * extent) : int((height + y) * extent), 
                    int(x * extent) : int((width + x) * extent)
                ], 
            probabilityFrames
        )))
    
    unnormalized = np.array(list(map(lambda frame : 
            (spatialStep ** 2) * math.sum(math.sum(frame)), 
            cutFrames
        )))
    return unnormalized / normalizationValues, cutFrames

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

