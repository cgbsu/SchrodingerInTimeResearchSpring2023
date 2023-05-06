from libschrodinger import *

MatrixType = np.ndarray
SparseMatrixType = spsparse.spmatrix
MatrixSolverFunctionType = Callable[[ComputationalProfile, SparseMatrixType, MatrixType], MatrixType] 

def asNumPyArray(array) -> np.array: 
    isCuPyArray = (type(array) is cp.ndarray) or (type(array) is cp.array)
    return array.get() if isCuPyArray else array
