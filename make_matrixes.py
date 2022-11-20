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

        for N in range(2, 6, 1): # искусственно создаем строку нулей
            M = np.random.randint(low = 0, high = 101, size=(N,N))
            k = np.random.randint(0,N)
            for i in range(N):
                M[k][i] = 0
            M = M.tolist()
            matrixes.append(json.dumps(M))

        for N in range(2, 6, 1): # искусственно создаем столбец нулей
            M = np.random.randint(low = 0, high = 101, size=(N,N))
            k = np.random.randint(0,N)
            for i in range(N):
                M[i][k] = 0
            M = M.tolist()
            matrixes.append(json.dumps(M))
        json.dump(matrixes, f)

# make_scenario()
def open_scenarios():
    with open('scenarios.txt', 'rt') as f:
        matrixes = json.load(f)
    pipeline = list(map(lambda x : np.array(json.loads(x), dtype = 'float64'), matrixes))
    return pipeline

def open_scenarios_v():
    with open('right_vectors.txt', 'rt') as f:
        right_vectors = json.load(f)
    pipeline_v = list(map(lambda x : np.array(json.loads(x), dtype = 'float64'), right_vectors))
    return pipeline_v
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

def make_right_vectors():
    def make_vector(N):
        M = 50 - np.random.rand(N, 1) * 100
        M = M.tolist()
        right_vectors.append(json.dumps(M))

    with open('right_vectors.txt', 'wt') as f:
        right_vectors = []

        for N in range(2, 6, 1): # дампим матрицы вещественных чисел
            M = 50 - np.random.rand(N,1)*100
            M = M.tolist()
            right_vectors .append(json.dumps(M))
        for _ in range(3):
            for N in range(2, 6, 1): # дампим матрицы целых чисел
                make_vector(N)
        json.dump(right_vectors , f, indent=4)

def sle_cases(): # разбить СЛАУ на решаемые и нерешаемые
    matrixes = open_scenarios()
    right_vectors = open_scenarios_v()
    correct_cases = []
    singular_cases = []

    for system in list(zip(matrixes, right_vectors)):
        matrix, vector = system
        if np.linalg.det(matrix) != 0:
            correct_cases.append(system)
        else:
            singular_cases.append(system)
    return correct_cases, singular_cases

def dump_sle_cases(): # задампить решаемые СЛАУ и нерешаемые СЛАУ по разным файлам
    correct_cases, singular_cases = sle_cases()
    with open('correct_cases.txt', 'wt') as f:
        correct_cases_json = list(map(lambda x : [x[0].tolist(), x[1].tolist()], correct_cases))
        json.dump(correct_cases_json, f)

    with open('singular_cases.txt', 'wt') as f:
        singular_cases_json = list(map(lambda x : [x[0].tolist(), x[1].tolist()], singular_cases))
        json.dump(singular_cases_json, f)

def open_sle(sle_type = 'correct'):
    if sle_type == 'correct':
        with open('correct_cases.txt', 'rt') as f:
            sle = json.load(f)
            pipeline_sle = list(
                map(
                    lambda x : [
                        np.array(x[0], dtype='float64'),
                        np.array(x[1], dtype='float64')
                    ], sle
                )
            )

    if sle_type == 'singular':
        with open('singular_cases.txt', 'rt') as f:
            sle = json.load(f)
            pipeline_sle = list(
                map(
                    lambda x : [
                        np.array(x[0], dtype='float64'),
                        np.array(x[1], dtype='float64')
                    ], sle
                )
            )

    return pipeline_sle
