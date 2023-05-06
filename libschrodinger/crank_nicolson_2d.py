from libschrodinger import *

# TODO: 
# - Figure out why potentials need to be so much higher 
#       (are potentials "trapping" the wave function?) 
#       than the demo this is built apon
# - Add "delay" to potentials in time dimension
# - Make dimensions heterogenus?
# - Figure out best way to encode potentials

class SimulationResults: 
    def __init__(self, waveFunction : MeshGrid): 
        self.waveFunctions = waveFunction if type(waveFunction) is list else [waveFunction]

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

    def compute(
                self, 
                unknownStepMatrix : SparseMatrixType, 
                knownStepMatrix : SparseMatrixType, 
                log : PerformenceLogType, 
                logFunction : LogFunctionType, 
                matrixSolveMethod : MatrixSolverFunctionType
            ) -> Tuple[MatrixType, MatrixType]: 
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
        nextWaveFunction = matrixSolveMethod(
                    self.profile, 
                    unknownStepMatrix, 
                    independantTerms, 
            )
        if self.profile.fineGrainedLog == True: 
            logFunction(log, "Fine Grain: Solve for nextWaveFunction")
        nextWaveFunction = nextWaveFunction.reshape(tuple([self.grid.pointCount - 2] * self.dimensions),)
        if self.profile.fineGrainedLog == True: 
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
                logFunction = None, 
                matrixSolveMethod : MatrixSolverFunctionType = None
            ): 
        math = self.math
        logFunction = logFunction if logFunction else self.profile.logFunction
        log = log if log else {}
        matrixSolveMethod = matrixSolveMethod if matrixSolveMethod \
                else self.profile.defaultMatrixSolveMethod 
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
            self.compute(unknownStepMatrix, knownStepMatrix, log, logFunction, matrixSolveMethod)
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

