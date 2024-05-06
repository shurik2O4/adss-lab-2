#!/usr/bin/env python3

from utils import *
from simplex import *
import numpy as np

# Get input
Z, limit = input_limit_equation("Z=")

# If the limit is max, invert
if limit == "max":
    Z = -Z
else:
    # When assembling a matrix, coefficients are inverted
    # Since max Z = -min Z', we can just set change limit to max
    limit = "max"

equations, constants = input_inequality_matrix("Boundaries:")

j = float(input("j="))
v = 0.

print_matrix(assemble_matrix(equations, constants, Z, limit, v), "Simplex matrix:")

# Solve the matrix (reference solution)
matrix = simplex(equations, constants, Z, limit, v, j)

print_matrix(matrix, "Matrix:")