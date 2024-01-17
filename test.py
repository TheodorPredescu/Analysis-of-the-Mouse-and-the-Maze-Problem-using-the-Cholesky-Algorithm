import numpy as np

def generate_matrix(nr):
    matrix = [[0.0] * nr for _ in range(nr)]

    x = 1
    sum_linii_precedente = 0
    ok = False
    vector = []

    # print("Initial Matrix:")
    # for row in matrix:
    #     print(" ".join(map(str, row)))

    # print("************************")
    while not ok:
        for i in range(1, x + 1):
            nr_descendenti = 0
            if i + sum_linii_precedente <= nr:
                if nr >= i + sum_linii_precedente + x:
                    nr_descendenti += 1
                if nr >= i + sum_linii_precedente + x + 1:
                    nr_descendenti += 1
            else:
                ok = True
                break

            if nr_descendenti != 0:
                if nr_descendenti >= 1:
                    matrix[i + sum_linii_precedente - 1][i + sum_linii_precedente - 1 + x] = 1.0
                    matrix[i + sum_linii_precedente - 1 + x][i + sum_linii_precedente - 1] = 1.0
                if nr_descendenti >= 2:
                    matrix[i + sum_linii_precedente - 1][i + sum_linii_precedente - 1 + x + 1] = 1.0
                    matrix[i + sum_linii_precedente - 1 + x + 1][i + sum_linii_precedente - 1] = 1.0
        sum_linii_precedente += x
        x += 1
        vector.append(sum_linii_precedente)
    # print(vector)
    x = x -1 
    sum_linii_precedente = sum_linii_precedente - x
    b = np.zeros(nr)
    if  sum_linii_precedente == nr: 
        for i in range(nr - x + 1, nr):
            b[i] = 1
    else:
        for i in range(sum_linii_precedente, nr):
            b[i] = 1
    return np.array(matrix), np.array(b)

i = int(input("Give nr: "))
alfa = 1/4
A,b = generate_matrix(i)
b = b*alfa*2
Im = np.identity(A.shape[0])
A = Im - A*alfa
print(A)
print(b)

x = np.linalg.solve(A, b)
print()
print(x)