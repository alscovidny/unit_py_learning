import numpy as np
from make_matrixes import open_sle
from functools import reduce

class SingularMatrixException(Exception):
    pass

ACCURACY = 10

def solve_MP(A, b, accuracy=6): #via Moore-Penrose pseudo inverse
    if np.linalg.det(A) == 0:
        raise SingularMatrixException('Using SLE with singular matrix')
    return np.round(np.linalg.inv(A.T@A)@A.T@b, decimals=accuracy)

def solve_LU(A, b, accuracy=6):
    if np.linalg.det(A) == 0:
        raise SingularMatrixException('Using SLE with singular matrix')

    if A.shape[0] != A.shape[1]:
        return None
    N = A.shape[0]
    L = np.eye(N)  # диагональная единичная матрица
    U = np.zeros(A.shape)

    for i in range(N):
        for j in range(N):
            if i <= j:
                U[i][j] = A[i][j] - np.sum([L[i][k] * U[k][j] for k in range(i)])
            if i > j:
                L[i][j] = (A[i][j] - np.sum([L[i][k] * U[k][j] for k in range(j)])) / U[j][j]
    # 1.решение Ly = b методом прямой подстановки
    y = np.zeros((N, 1))
    for i in range(N):
        if i == 0:
            y[0] = b[0] / L[0][0]
        else:
            res = b[i]
            for k in range(i):
                res = res - L[i][k] * y[k]
            y[i] = res / L[i][i]

    # 2.решение Ux = y методом обратной подстановки
    x = np.zeros((N, 1))
    for j in range(N - 1, -1, -1):
        if j == N - 1:
            x[j] = y[j] / U[j][j]
        else:
            res = y[j]
            for k in range(N - 1, j, -1):
                res = res - U[j][k] * x[k]
            x[j] = res / U[j][j]
    return np.round(x, decimals=accuracy)

def is_correct_LU(cases = open_sle(sle_type='correct'), accuracy=ACCURACY):
    is_correct_solution_LU = []
    for matrix, vector in cases:
        res_LU = solve_LU(matrix, vector, accuracy)
        res_numpy = np.round(np.linalg.solve(matrix, vector), decimals=accuracy)
        is_correct_solution_LU.append(bool(reduce(lambda x, y: x & y, res_LU == res_numpy)))
    return is_correct_solution_LU

def is_correct_MP(cases=open_sle(sle_type='correct'), accuracy=ACCURACY):
    is_correct_solution_MP = []
    for matrix, vector in cases:
        res_MP = solve_LU(matrix, vector, accuracy)
        res_numpy = np.round(np.linalg.solve(matrix, vector), decimals=accuracy)
        is_correct_solution_MP.append(bool(reduce(lambda x, y: x & y, res_MP == res_numpy)))
    return is_correct_solution_MP
