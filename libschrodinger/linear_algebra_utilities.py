from libschrodinger.computational_profile import *
from libschrodinger.linear_algebra_utilities import *
import numpy as np
import scipy.sparse as spsparse

MatrixType = np.ndarray
SparseMatrixType = spsparse.spmatrix

def solveMatrixStandard(
            profile : ComputationalProfile, 
            operator : SparseMatrixType, 
            independantTerms : MatrixType
        ) -> MatrixType: 
    sparse = profile.sparse
    linalg = profile.linalg
    return linalg.spsolve(sparse.csr_matrix(operator), independantTerms)

def solveMatrixApproximate(
            profile : ComputationalProfile, 
            operator : SparseMatrixType, 
            independantTerms : MatrixType, 
            tolerence : float
        ) -> MatrixType: 
    sparse = profile.sparse
    linalg = profile.linalg
    return linalg.cg(operator, independantTerms, x0 = None, tol = tolerence)[0]

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
