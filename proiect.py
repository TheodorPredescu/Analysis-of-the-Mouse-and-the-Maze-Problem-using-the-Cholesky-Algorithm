import numpy as np
import matplotlib.pyplot as plt
import math
import os
import time


def create_matrix_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        matrix_data = []
        for i, line in enumerate(lines):
            # Stop reading when encountering a blank line
            if not line.strip():
                break
            if i == 0:
                continue

            matrix_data.append(line.strip().split())

        rows = len(matrix_data)
        cols = len(matrix_data[0])

        matrix = [[float(matrix_data[i][j]) for j in range(cols)] for i in range(rows)]

        return np.array(matrix)

    except FileNotFoundError:
        print(f"File not found_create: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def extract_first_value(file_path):
    try:
        with open(file_path, 'r') as file:
            first_line = file.readline().strip()
            first_value = float(first_line)
            return first_value

    except FileNotFoundError:
        print(f"File not found_first_value: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def extract_b(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        blank_line_found = False
        for i in range(len(lines)):
            # Set blank_line_found to True when encountering a blank line
            if not lines[i].strip():
                blank_line_found = True
                continue

            # If a blank line has been found, extract and return the content of the next line
            if blank_line_found:
                line_content = lines[i].strip().split()
                return np.array([float(value) for value in line_content])

        return None

    except FileNotFoundError:
        print(f"File not found_b: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def get_b(rows):
    row_input = input(f"Enter the elements that have acces to the destination(in a line enter {rows} elements, the pozition beeing the intersection that you refer to and 0 or 1 if it does or does not have acces to the destination): ")
    row = [float(x) for x in row_input.split()]
    return np.array(row)

def LTRIS( L, b):
    x = b
    start_line = 0
    for i in range(len(L)):
        for j in range(start_line, i):
            # ceva e suspect; ==0 sau != 0 sau foarte aproape de adevar, dar doar ==0 teoretic e corecta
            if L[i][j] == 0:
                start_line += 1
                # print(f"i = {i}\nstart line = {start_line}\n\n")
            x[i] -= L[i][j]*x[j]
        x[i] = x[i]/L[i][i]
    return x

def UTRIS( U, b):
    x = b
    stop_line = len(b)
    for i in range(len(U) - 1, -1, -1):
        for j in range(i + 1, stop_line):
            # aceeasi posibila problema
            if U[i][j] == 0:
                stop_line -= 1    
            x[i] -=U[i][j]*b[j]
        x[i] = x[i]/U[i][i]
    return x

def CHOL(A):
    n = len(A)
    L = np.zeros_like(A, dtype=float)
    
    for k in range(n):
        sum1 = 0
        for j in range(k):
            sum1 += L[k, j] ** 2
        L[k, k] = math.sqrt(A[k, k] - sum1)
        
        for i in range(k + 1, n):
            sum2 = 0
            for j in range(k):
                sum2 += L[i, j] * L[k, j]
            L[i, k] = (A[i, k] - sum2) / L[k, k]
    # print(L)
    return L

def generate_matrix(nr):
    matrix = [[0.0] * nr for _ in range(nr)]

    x = 1
    sum_linii_precedente = 0
    ok = False

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

def direct_calcul(nr):
    A,b = generate_matrix(nr)
    identity_matrix = np.identity(A.shape[0])
    A = identity_matrix - A /4
    b = b/2
    A = CHOL(A)
    b = LTRIS(A,b)
    At = np.transpose(A)
    b = UTRIS(At,b)

def kaczmarz(A, b, max_iter=1000, tol=1e-6):
    m, n = A.shape
    x = np.zeros(n)

    for iteration in range(max_iter):
        for i in range(m):
            if np.abs(np.dot(A[i, :], x) - b[i]) > tol:
                alpha = (b[i] - np.dot(A[i, :], x)) / np.linalg.norm(A[i, :]) ** 2
                x = x + alpha * A[i, :]

    return x

def kaczmarx_random(nr):
    A,b = generate_matrix(nr)
    identity_matrix = np.identity(A.shape[0])
    A = identity_matrix - A /4
    b = b/2
    x = kaczmarz(A, b)
    print(x)



ok = True
while ok:
    os.system('cls' if os.name == 'nt' else 'clear')
    print("1. Fisier\n2. Tastatura\n3. Speed time\n4. Exit\n\n")
    option = int(input("optiune:"))
    if option == 4:
        os.system('cls' if os.name == 'nt' else 'clear')
        break
    elif option == 3:
        while True:
            number_plots = int(input("Introdu nr de rulari(minim 10): "))
            print("Va rugam astepati...")
            vector_CHOL = []
            vector_kaczmarx = []
            if number_plots >=10:
                for i in range (10, number_plots + 1):
                    
                    start_time = time.time()
                    direct_calcul(i)
                    stop_time = time.time()
                    execution_time = stop_time - start_time
                    vector_CHOL.append(execution_time)

                    start_time_k = time.time()
                    kaczmarx_random(i)
                    stop_time_k = time.time()
                    execution_time_k = stop_time_k - start_time_k
                    vector_kaczmarx.append(execution_time_k)
                    
                plt.plot(vector_CHOL, label='Vector CHOL')
                plt.plot(vector_kaczmarx, label='Vector kaczmarx')
                plt.xlabel('Nr de intersectii')
                plt.ylabel('Durata')
                plt.legend()
                plt.show()
                break
            else:
                print("Am zis mai mare decat 10!\n")

    elif option == 1:
    # pentru input din fisier
        print("The file should have the following format:\n\n\nNUMBER OF CONECTIONS\nMATRIX OF CONECTIONS\n(free line)\nTHE VECTOR CONECTIONS WITH FOOD\n\n")
        file_name = input('Give the file name (txt file only): ')
        file_path = 'C:\\Users\\prede\\Desktop\\facultate\\anul2_sem1\\MN\\project\\input\\' + file_name + '.txt'
        # file_path = 'input\\'+ file_name + '.txt'
        alfa = extract_first_value(file_path)
        # print(alfa)
        A = create_matrix_from_file(file_path)
        # print(A)
        b = extract_b(file_path)
        # print(b)
    elif option == 2:
        # input tastatura
        os.system('cls' if os.name == 'nt' else 'clear')
        print("Labirintul e de forma triunghiulara - exemplu mai jos:\n")
        print("           / \\\n          /   \\\n         / \ / \\\n        /   1   \\\n       / \ / \ / \\\n      /   2   3   \\\n     /   / \ / \   \\\n         mancare\n")
        nr = int(input("Introduceti nr de elemente: "))
        A,b = generate_matrix(nr)
        # alfa = float(input("Da alfa: "))
        alfa = 4.0
    else:
        print("optiune gresita!")
    if option == 2 or option == 1 :
        alfa = 1/alfa
        b = b * alfa * 2
        identity_matrix = np.identity(A.shape[0])
        A = identity_matrix - alfa * A
        
        # print(A)
        # print()
        # print(b)

        xx = np.linalg.solve(A, b)

        A = CHOL(A)
        # print("***********************")
        # print(A)
        # print("***********************")
        b = LTRIS(A, b)
        At = np.transpose(A)
        b = UTRIS(At,b)
        print("\nThe probabilities are:\n")
        cnt = 0
        for x in b:
            cnt = cnt + 1
            print(f"Intersection {cnt}: {x}")
        print(f"\nResult of \"np.linalg.solve\":\n")
        cnt = 0
        for x in xx:
            cnt = cnt + 1
            print(f"p{cnt}: {x}")
    input("\n(Press to -ENTER- continue)")

# Assuming you have a 10000x10000 matrix representing accuracy values
#accuracy_matrix = np.random.rand(10000, 10000)  # Replace this with your accuracy data

def plotFunction(A):
    
    # Plotting accuracy matrix as a heatmap
    plt.figure(figsize=(8, 6))
    # plt.imshow(accuracy_matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Accuracy')
    plt.title('Accuracy of Probability Guessing')
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    plt.show()

#buna
# buna si tie

