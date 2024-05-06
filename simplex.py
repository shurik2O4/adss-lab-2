from typing import Iterable
import numpy as np
from modified_jordan import jordan_elimination, columns, rows
from utils import print_matrix

def first_negative(arr: Iterable) -> tuple[int, float]:
    return filter(lambda x: x[1] < 0, enumerate(arr)).__next__()

def min_non_negative(arr: Iterable) -> tuple[int, float]:
    return min([(i, v) for i, v in enumerate(arr)], key=lambda x: x[1] if x[1] >= 0 else np.inf)

def assemble_matrix(equation: np.ndarray, constants: np.ndarray, Z: np.ndarray, limit: str, v: float) -> np.ndarray:
    # [ EQUATION ] [c]
    # [ EQUATION ] [c]
    # [ EQUATION ] [c]
    # [ EQUATION ] [c]
    # [    Z     ] [v]

    # Put constants on the right
    copy = np.hstack((equation, constants))

    bottom = np.append(Z, v)

    # Add the bottom row
    return np.vstack((copy, bottom))

def disassemble_matrix(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    # Separate bottom row
    Z = matrix[-1, :-1]
    v = matrix[-1, -1]

    # Remove bottom row
    sliced = matrix[:-1]

    # Separate constants into one column array
    constants = np.array([np.array([i]) for i in sliced[:, -1]])

    # Remove constants
    equation = sliced[:, :-1]

    return equation, constants, Z, v

# def 

def div_column_by_constants(equation: np.ndarray, column: int, constants: np.ndarray) -> np.ndarray:
    return np.array([c[0] for c in constants]) / equation[:, column]

def collect_solution(rows: rows, constants: np.ndarray, solution: np.ndarray) -> None:
    # If X is in the row, add the constant to the solution
    for i, (r, xi) in enumerate(rows):
        if r == 'x':
            solution[xi - 1] = constants[i][0]

def simplex(equation: np.ndarray, constants: np.ndarray, Z: np.ndarray, limit: str, v: float, j: float) -> tuple[np.ndarray, np.ndarray]:
    step = 0
    row_names = [('y', i + 1) for i in range(equation.shape[0])]
    column_names = [('x', i + 1) for i in range(equation.shape[1])]

    ref_solution = np.array([0] * equation.shape[1])
    optimal_solution = np.array([0] * equation.shape[1])

    # Look for a negative value in the constants column
    while np.any(constants < 0):
        constants_row_i, _ = first_negative(constants)

        try:
            # Find a negative value in the row
            # col = np.where(equation[constants_row_i] < 0)[0][0]
            col, _ = first_negative(equation[constants_row_i])
        except IndexError as e:
            raise ValueError(f"Boundaries are contradictory: {e}")

        # Divide each element in the column by the element in the constants
        div_result = div_column_by_constants(equation, col, constants)
        
        # Find the minimum positive value
        row, _ = min_non_negative(div_result)

        row_names, column_names, matrix = jordan_elimination(row_names, column_names, assemble_matrix(equation, constants, Z, limit, v), row, col)
        equation, constants, Z, v = disassemble_matrix(matrix)

    collect_solution(row_names, constants, ref_solution)

    while np.any(Z < 0):
        col, _ = first_negative(Z)

        div_result = div_column_by_constants(equation, col, constants)
        row, _ = min_non_negative(div_result)

        row_names, column_names, matrix = jordan_elimination(row_names, column_names, assemble_matrix(equation, constants, Z, limit, v), row, col)
        equation, constants, Z, v = disassemble_matrix(matrix)

        step += 1
        print_matrix(matrix, f"Step {step}:")

    collect_solution(row_names, constants, optimal_solution)

    print(f"Reference solution: X: ({'; '.join([str(i) for i in ref_solution])})")
    print(f"Optimal solution: X: ({'; '.join([str(i) for i in optimal_solution])})")

    return ref_solution, optimal_solution