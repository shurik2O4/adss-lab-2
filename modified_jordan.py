from typing import Callable
from utils import input_matrix, print_exc_and_exit, print_matrix
import numpy as np

rows = list[tuple[str,int]]
columns = list[tuple[str,int]]

# def invert(matrix: np.ndarray, verbose: bool = True, error_handler: Callable = print_exc_and_exit) -> tuple[int, np.ndarray]:
#     r = 0
#     copy = matrix.copy()
#     # Since we're inverting diagonally, 
#     # we can only go as far as the smallest dimension
#     for i in range(min(matrix.shape[0], matrix.shape[1])):
#         try:
#             copy = jordan_elimination(copy, i, i)
#             if verbose:
#                 print_matrix(copy, f"Step {r + 1}:")
#             r += 1
#         except ValueError as e:
#             error_handler(e)
#     return r, copy

def jordan_elimination(rows: rows, columns: columns, matrix: np.ndarray, r: int, s: int) -> tuple[rows, columns, np.ndarray]:
    if matrix[r][s] == 0: raise ValueError(f"Matrix value at ({r},{s}) is 0")

    copy = matrix.copy()
    # 3: Invert the column sign
    copy[:, s] = -copy[:, s]

    # 1: Set the element to 1
    copy[r][s] = 1

    # 2: Don't change row elements
    # 4: Calculate other elements
    # r, i - row
    # s, j - column
    for i in range(copy.shape[0]):
        for j in range(copy.shape[1]):
            # Don't touch the row and column
            if i != r and j != s:
                copy[i][j] = matrix[i][j] * matrix[r][s] - matrix[i][s] * matrix[r][j]

    # Extra: swap the row and column names
    columns[s], rows[r] = rows[r], columns[s]

    # 5: Divide the new matrix by the element
    return (rows, columns, copy / matrix[r][s])

#################
# def jordan_elemination_verbose(matrix: np.ndarray, r: int, s: int) -> np.ndarray:
#     if matrix[r][s] == 0:
#         raise ValueError(f"Matrix value at ({r},{s}) is 0")
    
#     copy = matrix.copy()
#     # 2: Invert the row sign
#     copy[r] = -copy[r]

#     print("2: Invert the row sign")
#     print(copy)

#     # 1: Set the element to 1
#     copy[r][s] = 1

#     print("1: Set the element to 1")
#     print(copy)

#     # 3: Don't change column elements
#     # 4: Calculate other elements
#     # r, i - row
#     # s, j - column
#     for i in range(copy.shape[0]):
#         for j in range(copy.shape[1]):
#             # Don't touch the row and column
#             if i != r and j != s:
#                 copy[i][j] = matrix[i][j] * matrix[r][s] - matrix[i][s] * matrix[r][j]
#                 print(f"4: [{i}, {j}] = {matrix[i][j]} * {matrix[r][s]} - {matrix[i][s]} * {matrix[r][j]} = {copy[i][j]}")
#             else:
#                 print(f"4: [{i}, {j}] = {matrix[i][j]} (Skipped)")
    
#     print("4: Calculate other elements")
#     print(copy)

#     # 5: Divide the new matrix by the element
#     copy = copy / matrix[r][s]

#     print("5: Divide the new matrix by the element")
#     print(copy)

#     return copy
###################

# print("Jordan Elimination:\n", jordan_elemination(matrix, int(input("X: ")), int(input("Y: "))), sep="")

# step = jordan_elemination(matrix, 0, 0)
# step = jordan_elemination(step, 1, 1)
# step = jordan_elemination(step, 2, 2)
# step = jordan_elemination(step, 3, 3)
# step = np.round(step, 2)
# print("Jordan Elimination:\n", step, sep="")

# print("Jordan Elimination:\n", jordan_elemination(matrix, 0, 0), sep="")
# print("Jordan Elimination:\n", , sep="")



if __name__ == "__main__":
    # Test the function
    # matrix = input_matrix(print_back=True)
    # matrix = np.array([[-1.,  0.,  3., -2.,  1.,  3.],
    #                    [ 1., -1.,  0.,  1.,  1.,  3.],
    #                    [-1., -3.,  1.,  1., -1., -2.],
    #                    [-1.,  1.,  0.,  0.,  1.,  0.]])
    matrix = np.array([[-1.,  3.,  2., -3.,  2.,  5.],
                       [ 1., -4.,  1.,  2., -0.,  1.],
                       [-1.,  3., -1., -1.,  1.,  2.],
                       [-1.,  4., -1., -1.,  2.,  2.]])
    result = jordan_elimination(matrix, int(input("Row: ")) - 1, int(input("Col: ")) - 1)
    print_matrix(result, "Result:")