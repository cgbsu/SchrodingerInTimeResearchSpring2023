from typing import Tuple, List, Dict, Callable
from libschrodinger.computational_profile import *
from libschrodinger.linear_algebra_utilities import *
from libschrodinger.mesh_grid import MeshGrid
from libschrodinger.performence_profiling import *

MatrixSolverFunctionType = Callable[[ComputationalProfile, SparseMatrixType, MatrixType], MatrixType] 

class SimulationProfile(ComputationalProfile): 
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
                length : int = None, 
                logFunction : LogFunctionType = lambda log, stepLabel : None, 
                fineGrainedLog : bool = False, 
                defaultMatrixSolveMethod : MatrixSolverFunctionType = solveMatrixStandard
            ): 
        super().__init__(gpuAccelerated)
        assert (gpuAccelerated == False) if useDense == True else True
        assert (timeStep / spaceStep) <= courantNumber if courantWarning == True else True, \
                "Courant condition not satisfied! timeStep / spaceStep greater than Courant Number (usually 1)"
        self.grid = grid
        self.dimensions = self.grid.dimensions
        self.initialWaveFunctionGenerator = initialWaveFunctionGenerator
        self.potentialGenerator = potentialGenerator
        self.timeStep = timeStep
        self.spaceStep = spaceStep
        self.edgeBound = edgeBound
        self.useDense = useDense
        self.constantPotential = constantPotential 
        self.length = length
        self.logFunction = logFunction
        self.fineGrainedLog = fineGrainedLog
        self.defaultMatrixSolveMethod = defaultMatrixSolveMethod
