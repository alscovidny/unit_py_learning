import json
import numpy as np

def make_scenario():
    with open('scenarios.txt', 'wt') as f:
        matrixes = []

        for N in range(2, 6, 1): # дампим матрицы вещественных чисел
            M = 50 - np.random.rand(N,N)*100
            M = M.tolist()
            matrixes.append(json.dumps(M))

        for N in range(2, 6, 1): # дампим матрицы целых чисел
            M = 50 - np.random.randint(low = 0, high = 101, size=(N,N))
            M = M.tolist()
            matrixes.append(json.dumps(M))

        for N in range(2, 6, 1):
            M = np.random.randint(low = 0, high = 101, size=(N,N))
            k = np.random.randint(0,N)
            for i in range(N):
                M[k][i] = 0
            M = M.tolist()
            matrixes.append(json.dumps(M))

        for N in range(2, 6, 1):
            M = np.random.randint(low = 0, high = 101, size=(N,N))
            k = np.random.randint(0,N)
            for i in range(N):
                M[i][k] = 0
            M = M.tolist()
            matrixes.append(json.dumps(M))
        json.dump(matrixes, f, indent=4)

# make_scenario()
def open_scenarios():
    with open('scenarios.txt', 'rt') as f:
        matrixes = json.load(f)
        #
        # for M in matrixes:
        #     mtx = np.array(json.loads(M), dtype='float64')
        #
    pipeline = list(map(lambda x : np.array(json.loads(x), dtype = 'float64'), matrixes))
    return pipeline

print(open_scenarios())

def gauss_det(a):
    a = a.copy()
    if a.shape[0] != a.shape[1]:
        return None
    else:
        N = a.shape[0]

    perm = 0
    for j in range(N):

        # перестановка, если ajj = 0:
        if a[j][j] == 0:
            for i in range(j + 1, N):
                if a[i][j] != 0:
                    a[[i, j]] = a[[j, i]]
                    perm += 1
                    break
            else:
                return 0  # окончание выполнения кода
                break

        for k in range(j + 1, N):  # смотрим строки от j+1 до N
            coef = a[k][j] / a[j][j]
            for i in range(N):
                a[k][i] = a[k][i] - coef * a[j][i]

    det = 1 if perm // 2 or perm == 0 else -1

    for j in range(N):
        det = det * a[j][j]
    return det
