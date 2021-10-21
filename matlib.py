"""
matlib.py

Put any requested function or class definitions in this file.  You can use these in your script.

Please use comments and docstrings to make the file readable.
"""
from scipy import linalg as la
import numpy as np
from matplotlib import style
style.use('dark_background')
from numba import jit


# Problem 0

# Part A
def solve_chol(A, b):
    # Note that the output of cholesky() is an upper triangular matrix so we need to take
    # its transpose to make everything consistent. It is of course possible to just use 
    # the upper triangular matrix and call it R.
    L = la.cholesky(A).T
    # We first solve for "intermediate" vector y = L^T * x
    y = la.solve_triangular(L, b, lower = True)
    # We then solve for x. Note x = L^(-T) * y
    x = la.solve_triangular(L.T, y, lower = False)
    return x

# Part B
def matrix_pow(A,n):
    # EVD for symmetric matrix
    D, Q = la.eigh(A)
    # Perform elementwise power
    D = np.power(D,n)
    # we create a matrix whose diagonal entries are D and return the result
    return (Q @ np.diag(D) @ Q.T)

# Part D
def abs_det(A):
    # Perform LU decomposition
    P,L,U = la.lu(A)
    # Return the absolute value of product of entries of U
    # We dont need to worry about P and L because P is just a permutation matrix which
    # has its determinant of -1 or 1, same as for L.
    return abs(np.product(np.diagonal(U)))

# Problem 1

# Part A
@jit(nopython = True)
def matmul_ijk(B, C):
    # Assume B is p x q, C is q x r
    p = B.shape[0]
    q = B.shape[1]
    r = C.shape[1]
    A = np.zeros((p,r))
    
    # We have three nested loops according to matrix multiplication formula
    for i in range(p):
        for j in range(r):
            for k in range(q):
                A[i,j] = A[i,j] + B[i,k] * C[k,j]
                
    return A

# Everything from here on is the same besides we change the order of loop
@jit(nopython = True)
def matmul_ikj(B,C):
    p = B.shape[0]
    q = B.shape[1]
    r = C.shape[1]
    A = np.zeros((p,r))
    
    for i in range(p):
        for k in range(q):
            for j in range(r):
                A[i,j] = A[i,j] + B[i,k] * C[k,j]
                
    return A

@jit(nopython = True)
def matmul_jik(B,C):
    p = B.shape[0]
    q = B.shape[1]
    r = C.shape[1]
    A = np.zeros((p,r))
    
    for j in range(r):
        for i in range(p):
            for k in range(q):
                A[i,j] = A[i,j] + B[i,k] * C[k,j]
                
    return A

@jit(nopython = True)
def matmul_jki(B,C):
    p = B.shape[0]
    q = B.shape[1]
    r = C.shape[1]
    A = np.zeros((p,r))
    
    for j in range(r):
        for k in range(q):
            for i in range(p):
                A[i,j] = A[i,j] + B[i,k] * C[k,j]
                
    return A

@jit(nopython = True)
def matmul_kij(B,C):
    p = B.shape[0]
    q = B.shape[1]
    r = C.shape[1]
    A = np.zeros((p,r))
    
    for k in range(q):
        for i in range(p):
            for j in range(r):
                A[i,j] = A[i,j] + B[i,k] * C[k,j]
                
    return A

@jit(nopython = True)
def matmul_kji(B,C):
    p = B.shape[0]
    q = B.shape[1]
    r = C.shape[1]
    A = np.zeros((p,r))
    
    for k in range(q):
        for j in range(r):
            for i in range(p):
                A[i,j] = A[i,j] + B[i,k] * C[k,j]
                
    return A

# Part(B)
@jit(nopython = True)
def matmul_blocked(B,C):
    n = B.shape[0]
    def isodd(n):
        return n%2
    if isodd(n):
        return matmul_ijk(B,C)
    elif n > 64:
        A = np.zeros((n,n))
        slices = (slice(0, n//2), slice(n//2, n))
        for I in slices:
            for J in slices:
                for K in slices:
                    A[I, J] = A[I, J] + matmul_blocked(B[I, K], C[K, J])
        return A
    else:
        return matmul_ijk(B,C)


    
# Part(C)
@jit(nopython = True)
def matmul_strassen(B,C):
    n = B.shape[0]
    
    def isodd(n):
        return n%2
    
    if isodd(n):
        return matmul_ijk(B,C)
    elif n > 64:
        A = np.zeros((n,n))
        # We here slice the matrix into smaller size block matrices
        s1 = slice(0, n//2)
        s2 = slice(n//2, n)
        B11, B12, B21, B22 = B[s1,s1], B[s1,s2], B[s2,s1], B[s2,s2]
        C11, C12, C21, C22 = C[s1,s1], C[s1,s2], C[s2,s1], C[s2,s2]
        # Here we define each M block
        M1 = matmul_strassen(B11 + B22, C11 + C22)
        M2 = matmul_strassen(B21 + B22, C11)
        M3 = matmul_strassen(B11, C12 - C22)
        M4 = matmul_strassen(B22, C21 - C11)
        M5 = matmul_strassen(B11 + B12, C22)
        M6 = matmul_strassen(B21 - B11, C11 + C12)
        M7 = matmul_strassen(B12 - B22, C21 + C22)
        # Construct result A
        A[s1,s1] = M1 + M4 - M5 + M7
        A[s1,s2] = M3 + M5
        A[s2,s1] = M2 + M4
        A[s2,s2] = M1 - M2 + M3 + M6
        return A
    else:
        return matmul_ijk(B,C)


# Problem 2
@jit(nopython = True)
def markov_matrix(n):
    A = np.zeros((n,n))
    for i in range(n):
        # two endpoints
        if i == 0:
            A[i,i] = A[i,i+1] = 0.5
        elif i == n-1:
            A[i,i-1] = A[i,i] = 0.5
        # the middle rows are the same in sense of off-diagonal and sup-diagonals are all 0.5
        else:
            A[i,i-1] = A[i,i+1] = 0.5
    return A
