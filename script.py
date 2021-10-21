"""
use this file to script the creation of plots, run experiments, print information etc.

Please put in comments and docstrings in to make your code readable
"""

"""
Note: I comment everything out because we are running into problems with submission
running time on GitHub.

from matlib import *
import matplotlib.pyplot as plt

# Problem 0

# Part(B)

import time
# We first define value of n. Making sure they are as type intger.
vals = np.round(np.logspace(1, np.round(np.log10(4000), decimals = 5), num=10)).astype(int)
# Initialzation of 2 empty strings will later be used to store time.
lu_time = []
chol_time = []
for n in vals:
    # We should make sure that A is spd
    A = np.random.rand(n,n)
    A = A @ A.T
    # Perform cholesky decomposition and record time
    cholt0 = time.time()
    B = la.cholesky(A)
    cholt1 = time.time()
    chol_time.append(cholt1-cholt0)
    # Perform lu decomposition and record time
    lut0 = time.time()
    B = la.lu(A)
    lut1 = time.time()
    lu_time.append(lut1 - lut0)
# Make plot    
plt.loglog(vals, chol_time)
plt.loglog(vals, lu_time)
plt.xlabel("n")
plt.ylabel("runtime")
plt.title("Cholesky vs LU")
plt.legend(["Cholesky", "LU"])
plt.savefig("chollu.png")

# Problem 1

# Part(A)
vals = np.round(np.logspace(1, np.round(np.log10(1000), decimals = 5), num=10)).astype(int)
ijk_time = []
ikj_time = []
jik_time = []
jki_time = []
kij_time = []
kji_time = []
dgemm_time = []
matmul_time = []
for n in vals:
    print(n)
    # We should make sure that A is spd
    B = np.array(np.random.rand(n,n),order = 'C')
    C = np.array(np.random.rand(n,n),order = 'C')
    
    ijkt0 = time.time()
    A = matmul_ijk(B,C)
    ijkt1 = time.time()
    ijk_time.append(ijkt1-ijkt0)
    print(1)
    ikjt0 = time.time()
    A = matmul_ikj(B,C)
    ikjt1 = time.time()
    ikj_time.append(ikjt1-ikjt0)
    print(2)
    jikt0 = time.time()
    A = matmul_jik(B,C)
    jikt1 = time.time()
    jik_time.append(jikt1-jikt0)
    print(3)
    jkit0 = time.time()
    A = matmul_jki(B,C)
    jkit1 = time.time()
    jki_time.append(jkit1-jkit0)
    print(4)
    kijt0 = time.time()
    A = matmul_kij(B,C)
    kijt1 = time.time()
    kij_time.append(kijt1-kijt0)
    print(5)
    kjit0 = time.time()
    A = matmul_kji(B,C)
    kjit1 = time.time()
    kji_time.append(kjit1-kjit0)
    print(6)
    matmult0 = time.time()
    A = np.matmul(B,C)
    matmult1 = time.time()
    matmul_time.append(matmult1-matmult0)
    print(7)
    dgemmt0 = time.time()
    A = la.blas.dgemm(1.0,B,C)
    dgemmt1 = time.time()
    dgemm_time.append(dgemmt0-dgemmt1)
    print(8)
    
plt.loglog(vals, ijk_time)
plt.loglog(vals, ikj_time)
plt.loglog(vals, jik_time)
plt.loglog(vals, jki_time)
plt.loglog(vals, kij_time)
plt.loglog(vals, kji_time)
plt.loglog(vals, matmul_time)
plt.loglog(vals, dgemm_time)
plt.xlabel("n")
plt.ylabel("runtime")
plt.title("8 Different Matrix Multiplication")
plt.legend(["Order ijk", "Order ikj", "Order jik", "Order jki", "Order kij", "Order kji", "np.matmul", "dgemm"])
plt.savefig("matmul.png")


# Part(B)
block_time = []
ijk_new_time = []

for i in range(6,11):
    print(i)
    # We should make sure that A is spd
    B = np.array(np.random.rand(2**i,2**i),order = 'C')
    C = np.array(np.random.rand(2**i,2**i),order = 'C')
    
    start = time.time()
    A = matmul_blocked(B,C)
    end = time.time()
    block_time.append(end-start)
    
    start = time.time()
    A = matmul_ijk(B,C)
    end = time.time()
    ijk_new_time.append(end-start)
    
plt.loglog([6,7,8,9,10], block_time)
plt.loglog([6,7,8,9,10], ijk_new_time)
plt.xlabel("n")
plt.ylabel("runtime")
plt.title("blocked vs ijk")
plt.legend(["matmul_blocked", "Order ikj"])
plt.savefig("blockedijk.png")

# Part(C)
strassen_time = []
block_time = []

for i in range(6,11):
    print(i)
    # We should make sure that A is spd
    B = np.array(np.random.rand(2**i,2**i),order = 'C')
    C = np.array(np.random.rand(2**i,2**i),order = 'C')
    
    start = time.time()
    A = matmul_strassen(B,C)
    end = time.time()
    strassen_time.append(end-start)
    
    start = time.time()
    A = matmul_blocked(B,C)
    end = time.time()
    block_time.append(end-start)
    
plt.loglog([6,7,8,9,10], block_time)
plt.loglog([6,7,8,9,10], strassen_time)
plt.xlabel("n")
plt.ylabel("runtime")
plt.title("blocked vs strassen")
plt.legend(["matmul_blocked", "matmul_strassen"])
plt.savefig("blockedstrassen.png")

# Problem 2

# Part(2)
n = 50
# We will use this empty matrix to store result for different t, column-wise
p_record = np.zeros((n,3))
# There are two ways to do this problem, since the form is p at time t is p_t = A*A*A*...*A*p, where it has t number of A
# We can either use p_t = A^t * p or we do t times of A*p. The former is a matrix multiplication and latter is a matrix-vector
# multiplication, which the latter one is faster. BUT, since A is symmetric and tri-diagonal, we can use eigenvalue decomposition
# which leads to A*A*A = U*D*Ut*U*D*Ut*U*D*Ut = U*D^3*Ut, and we can vectorize D to make it faster, so we will use that.

# Here we are defining the matrix A. Even though we are not using markov_matrix() function
# we defined above, but they are exactly the same.
diag = np.zeros(n)
diag[0] = diag[n-1] = 0.5
off_diag = [0.5] * (n-1)
# Eigenvalue decomposition of markov matrix A at size txt
D,Q = la.eigh_tridiagonal(diag,off_diag)
for t in [10,100,1000]:
    # Initialization of p
    p = np.zeros(n)
    # p[0] = 1
    p[0] = 1
    # D^t
    D_new = np.diag(np.power(D,t))
    # This is our new A
    A = Q @ D_new @ Q.T
    # The resulting p
    p_record[:,[10,100,1000].index(t)] = A @ p
    
from matplotlib.pyplot import figure
figure(figsize=(10, 8), dpi=120)

plt.imshow(p_record, cmap='hot', interpolation='nearest')
plt.title("Heatmap for Markov Process")
plt.xlabel("t = 10,100,1000")
plt.ylabel("position")
plt.show()


# Part(3)
# Note this function is defined specifically for the question, it takes input of time t
# and output the Euclidean norm between p[t] and narmalized eigenvector whose eigenvalue is the largest
def my_norm(t):
    diag = np.zeros(n)
    diag[0] = diag[n-1] = 0.5
    off_diag = [0.5] * (n-1)
    # Eigenvalue decomposition of markov matrix A at size txt
    D,Q = la.eigh_tridiagonal(diag,off_diag)
    max_index = np.argmax(D)
    v = Q[:, max_index]
    v = v / np.sum(v)
    # Initialization of p
    p = np.zeros(n)
    # p[0] = 1
    p[0] = 1
    D = np.diag(np.power(D,t))
    # This is our new A
    A = Q @ D @ Q.T
    # The resulting p
    return la.norm(A@p - v)

my_norm(1000)
my_norm(2000)
my_norm(10000)
my_norm(20000)
"""