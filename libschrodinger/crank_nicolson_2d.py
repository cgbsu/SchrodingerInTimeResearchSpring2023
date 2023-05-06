from enum import Enum
from dataclasses import dataclass
import numpy as np
import scipy.sparse as spsparse
import cupy as cp
import cupyx.scipy.sparse as cpsparse
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import numba
from numba import njit, jit
import scipy.sparse.linalg as spla
import cupyx.scipy.sparse.linalg as cpla
from datetime import timedelta
from time import monotonic
import sys

# TODO: 
# - Figure out why potentials need to be so much higher 
#       (are potentials "trapping" the wave function?) 
#       than the demo this is built apon
# - Add "delay" to potentials in time dimension
# - Make dimensions heterogenus?
# - Figure out best way to encode potentials

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

def asNumPyArray(array) -> np.array: 
    isCuPyArray = (type(array) is cp.ndarray) or (type(array) is cp.array)
    return array.get() if isCuPyArray else array

def printWithProgressBar(
            step, 
            timePoints : int, 
            progressBarLength : int = 100, 
            printProgress : bool = False, 
            showTotalTime : bool = False, 
            showStepTime : bool = False, 
            detailedProgress : bool = False
        ): 
    performenceStartTime = monotonic()
    performenceAverageStepTime = 1
    deltaProgress = 1
    lastProgressLength = 0
    messageLength = 1
    progressOffset : int = timePoints % progressBarLength
    if printProgress == True: 
        if detailedProgress == False: 
            sys.stdout.write("[" + ("=" * progressBarLength) + "]\n")
        sys.stdout.write("[")
        if detailedProgress == True: 
            sys.stdout.write("]")
    for ii in range(1, timePoints): 
        previousPerformenceTime = monotonic()
        step(ii)
        progress = round((ii / timePoints) * progressBarLength)
        if printProgress == True: 
            update = ((progress - lastProgressLength) // deltaProgress)
            if detailedProgress == True: 
                percent = " " + "{0:.3g}".format((ii / timePoints) * 100) + "%"
                frames = " (" + str(ii) + "/" + str(timePoints) + ")"
                performence = " {0:.3g} fps".format(1 / performenceAverageStepTime)
                remainingTime = ", est. {0:.3g}s remain".format((timePoints - ii) / (1 / performenceAverageStepTime))
                message = frames + percent + performence + remainingTime + "]"
                for jj in range(messageLength): 
                    sys.stdout.write("\b")
                    sys.stdout.flush()
            #if (progress - lastProgressLength) > deltaProgress: 
            sys.stdout.write("-" * update)
            if detailedProgress == True: 
                sys.stdout.write(message)
                messageLength = len(message)
            sys.stdout.flush()
            lastProgressLength += update
        performenceAverageStepTime = (performenceAverageStepTime \
                + (monotonic() - previousPerformenceTime)) / 2.0
    if printProgress == True: 
        sys.stdout.write("]\n")
        sys.stdout.flush()
    if showTotalTime == True: 
        print("Total Time: ", monotonic() - performenceStartTime)
    if showStepTime == True: 
        print("Frames Per Second: ", 1 / performenceAverageStepTime)

def performenceLog(log : Dict[str, List[float]], stepLabel : str): 
    if stepLabel not in log: 
        log[stepLabel] = []
    log[stepLabel].append(monotonic())

class SimulationProfile: 
    def __init__(
                self, 
                grid : MeshGrid, 
                initialWaveFunctionGenerator, 
                potentialGenerator, 
                timeStep, 
                spaceStep, 
                gpuAccelerated = False, 
                edgeBound = False, 
                useDense = False, 
                constantPotential = False, 
                courantNumber = 1.0, 
                courantWarning = True, 
                length = None, 
                logFunction = lambda log, stepLabel : None
            ): 
        assert (gpuAccelerated == False) if useDense == True else True
        assert (timeStep / spaceStep) <= courantNumber if courantWarning == True else True, \
                "Courant condition not satisfied! timeStep / spaceStep greater than Courant Number (usually 1)"
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
        self.useDense = useDense
        self.constantPotential = constantPotential 
        self.length = length
        self.logFunction = logFunction

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
            _centerDiagonalUnused, 
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
            _centerDiagonalUnused, 
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
            centerDiagonal, 
            shiftedInnerDiagonal, 
            innerDiagonal, 
            outerDiagonal
        ):
    math = simulator.math
    sparse = simulator.sparse
    unknownFactorCount = simulator.unknownFactorCount
    centerDiagonal = centerDiagonal[0] \
                + (scalar * 1j * simulator.timeStep / 2) \
                * currentPotential[1:-1, 1:-1].ravel()
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
        self.unknownFactorCount = (self.grid.pointCount - 2) ** self.dimensions
        self.diagonalLength = self.unknownFactorCount
        self.knownInnerDiagonal = -1 * self.math.ones(self.diagonalLength) \
                * self.stepConstants[0]
        if self.profile.edgeBound == True: 
            extent = int(self.math.sqrt(self.diagonalLength))
            self.knownInnerDiagonal.reshape((extent, extent)).T[-1] = 0
            self.knownInnerDiagonal = self.knownInnerDiagonal.reshape(self.diagonalLength)
        self.knownInnerDiagonalShifted = self.math.roll(self.knownInnerDiagonal, 1)
        self.knownOuterDiagonal = -1 * self.math.ones(self.diagonalLength) \
                * self.stepConstants[1]
        self.unknownInnerDiagonal = -1 * self.knownInnerDiagonal # These two just to save a bit of compute
        self.unknownOuterDiagonal = -1 * self.knownOuterDiagonal # time multiplying these large arrays by -1
        self.unknownInnerDiagonalShifted = -1 * self.knownInnerDiagonalShifted
        self.knownCenterDiagonal = self.math.ones(self.diagonalLength) \
                * (1 + self.math.sum(self.stepConstants * 2.0))
        self.unknownCenterDiagonal = self.math.ones(self.diagonalLength) \
                * (1 + self.math.sum(-self.stepConstants * 2.0))
        self.createStepMatrix = (
                        createDenseStepMatrixBounded \
                        if self.profile.edgeBound == True else createDenseStepMatrix
                ) if self.profile.useDense == True else createStepMatrix

    def createCurrentStepMatrix(self, currentPotential): 
        return self.createStepMatrix(
                self, 
                currentPotential, 
                1, 
                self.knownCenterDiagonal, 
                self.knownInnerDiagonalShifted, 
                self.knownInnerDiagonal, 
                self.knownOuterDiagonal
            )
    
    def createNextStepMatrix(self, nextPotential): 
        return self.createStepMatrix(
                self, 
                nextPotential, 
                -1, 
                self.unknownCenterDiagonal, 
                self.unknownInnerDiagonalShifted, 
                self.unknownInnerDiagonal, 
                self.unknownOuterDiagonal
            )

    def compute(self, unknownStepMatrix, knownStepMatrix, log, logFunction): 
        math = self.math
        sparse = self.sparse
        logFunction(log, "Started \"Compute\"")
        waveFunctionVector = self.waveFunctions[-1][1:-1, 1:-1].reshape(
                (self.diagonalLength, 1)
            )
        logFunction(log, "Reshaped Wave Function")
        independantTerms = knownStepMatrix @ waveFunctionVector 
        logFunction(log, "Matrix Multiplication")
        #independantTerms = math.matmul(knownStepMatrix, waveFunctionVector)
        #csrUnknownStepMatrix = sparse.csr_matrix(unknownStepMatrix)
        logFunction(log, "Fine Grain: sparse.csr_matrix")
        nextWaveFunction = self.linalg.cg(
                    unknownStepMatrix, 
                    independantTerms, 
                    x0 = None, 
                    tol = 1e-5#min(self.spaceStep, self.timeStep) ** 2
            )[0]
        logFunction(log, "Fine Grain: Solve for nextWaveFunction")
        nextWaveFunction = nextWaveFunction.reshape(tuple([self.grid.pointCount - 2] * self.dimensions),)
        logFunction(log, "Fine Grain: Reshape nextWaveFunction")
        logFunction(log, "Solved For Independant Terms and Reshaped")
        self.waveFunctions.append(math.pad(nextWaveFunction, 1))
        logFunction(log, "Appended Wave Function and Finished \"Compute\"")
        return self.waveFunctions[-1], self.potentials[-1]

    def simulateTime(self, maxTime : int, printProgress : bool = False): 
        math = self.math
        timeStep = self.timeStep
        timePoints = round(maxTime / timeStep)
        return self.simulate(timePoints, printProgress)

    def simulate(
                self, 
                timePoints : int, 
                printProgress : bool = False, 
                showTotalTime = False, 
                showStepTime = False, 
                detailedProgress = False, 
                progressBarLength : int = 100, 
                log = None, 
                logFunction = None
            ): 
        math = self.math
        logFunction = logFunction if logFunction else self.profile.logFunction
        log = log if log else {}
        #currentTime : float = timeStep
        logFunction(log, "Started Simulation")
        initialPotential = self.profile.potentialGenerator(self.grid, 0.0)
        logFunction(log, "Generated Initial Potential")
        if self.profile.constantPotential == True: 
            self.potentials = [initialPotential, initialPotential]
        else: 
            self.potentials = [
                    self.profile.potentialGenerator(self.grid, 0.0), 
                    self.profile.potentialGenerator(self.grid, self.timeStep)
                ]
        logFunction(log, "Generated Next Initial Potential")
        def step(stepIndex : int): 
            logFunction(log, "Starting \"Step\"")
            if self.profile.constantPotential == False: 
                self.potentials.append(
                        self.profile.potentialGenerator(self.grid, (timeIndex + 1) * self.timeStep)
                    )
                logFunction(log, "Generated Next Potential")
            knownStepMatrix = self.createCurrentStepMatrix(self.potentials[-2])
            logFunction(log, "Created knownStepMatrix")
            unknownStepMatrix = self.createNextStepMatrix(self.potentials[-1])
            logFunction(log, "Created unknownStepMatrix")
            self.compute(unknownStepMatrix, knownStepMatrix, log, logFunction)
            logFunction(log, "Computed and Finished \"Step\"")

        printWithProgressBar(
                step, 
                timePoints, 
                progressBarLength, 
                printProgress, 
                showTotalTime, 
                showStepTime, 
                detailedProgress
            )

        return log

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

def makeWavePacket(grid, startX, startY, spatialStep, sigma = 0.5, k = 15 * np.pi, math = np): 
    #unnormalized = math.exp((-1 / (2 * sigma ** 2)) * ((grid.x - startX) ** 2 + (grid.y - startY) ** 2)) \
            #* math.exp(-1j * k * (grid.x - startX))
    #totalProbability = math.sum(math.sqrt(math.real(unnormalized) ** 2 + math.imag(unnormalized) ** 2))
    #return unnormalized / totalProbability
    unnormalized = math.exp(-1 / 2 * ((grid.x - startX) ** 2 + (grid.y - startY) ** 2) / sigma ** 2) \
            * math.exp(1j * k * (grid.x - startX))
    return unnormalized

def constantPotentialRectangles(
            axis, 
            pointCount : int, 
            lengthRatios : List[float], 
            potentialRatios : List[float], 
            baseAlpha : float = .08, 
            color : str = "w", 
            zorder : int = 50
        ): 
    displayRectangles = []
    currentPosition = 0
    for ii in range(len(lengthRatios)): 
        xExtent = pointCount * lengthRatios[ii]
        displayRectangles.append(Rectangle(
                (currentPosition, 0), 
                xExtent, 
                pointCount, 
                color = color, 
                zorder = zorder, 
                alpha = potentialRatios[ii] * baseAlpha
            ))
        axis.add_patch(displayRectangles[-1])
        currentPosition += xExtent
    return displayRectangles

def animateImages(
            length : float, 
            images : List[np.array], 
            interval = 1, 
            minimumValue = None, 
            maximumValue = None, 
            lengthRatios = None, 
            potentialRatios = None, 
            baseAlpha : float = .08, 
            colorMap : str = "viridis"
        ): 
    animationFigure = plt.figure()
    animationAxis = animationFigure.add_subplot(xlim=(0, length), ylim=(0, length))
    animationFrame = animationAxis.imshow(
            asNumPyArray(images[0]), 
            extent=[0, length, 0, length], 
            vmin = minimumValue, 
            vmax = maximumValue, 
            zorder = 1, 
            cmap = colorMap
        )
    if lengthRatios and potentialRatios: 
        constantPotentialRectangles(
                animationAxis, 
                length, 
                lengthRatios, 
                potentialRatios, 
                baseAlpha = baseAlpha
            )
    def animateFrame(frameIndex): 
        animationFrame.set_data(asNumPyArray(images[frameIndex]))
        animationFrame.set_zorder(1)
        return animationFrame,
    animation = FuncAnimation(
            animationFigure, 
            animateFrame, 
            interval = interval, 
            frames = np.arange(0, len(images), 2), 
            repeat = True, 
            blit = 0
        )
    return animation

def totalProbabilityInRegion( # TODO: Find an AVERAGE normalization value to use for the whole thing, then use that.
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
    normalizationValues = math.array(list(map(lambda frame : 
            (spatialStep ** 2) * math.sum(math.sum(frame)), 
            probabilityFrames
        )))
    cutFrames = math.array(list(map(lambda frame : 
            frame[
                    int(y * extent) : int((height + y) * extent), 
                    int(x * extent) : int((width + x) * extent)
                ], 
            probabilityFrames
        )))
    
    unnormalized = math.array(list(map(lambda frame : 
            (spatialStep ** 2) * math.sum(math.sum(frame)), 
            cutFrames
        )))
    return unnormalized / normalizationValues, cutFrames

