# define symbolic variables
import sympy
import sympy as sp
from sympy import simplify

# g = sp.Symbol('g')
# b = sp.Symbol('b')
g = .5
b = .25
h = sp.Symbol('h')
M = sp.Symbol('M')
# C = sp.Symbol('C')
C = 0
K = sp.Symbol('K')
m = sp.Symbol('m')
s = sp.Symbol('s')

A = sp.matrices.Matrix([
    [2 + s * h, 0, -m * h],
    [h * g, h * g * K, M + h * g * C],
    [h ** 2 * b, M + h ** 2 * b * K, h ** 2 * b * C]
])

B = sp.matrices.Matrix([
    [2 - s * h, 0, m * h],
    [h * (g - 1), h * (g - 1) * K, M + h * (g - 1) * C],
    [h ** 2 * (b - .5), M + h ** 2 * (b - .5) * K, h * M + h ** 2 * (b - .5) * C]
])

# sp.pprint(A)
# sp.pprint(B)

C = A.inv() * B * A.det() / M

C = simplify(C)
# set pprint width
sp.pprint(C, num_columns=200)

D = simplify(C.eigenvals())

# sp.pprint(D, num_columns=200)

F = C[0, 0]

w = sympy.Symbol('w')

DD = sp.matrices.Matrix([
    [w, C[0, 1], C[0, 2]],
    [C[1, 0], w, C[1, 2]],
    [C[2, 0], C[2, 1], w]
])

print(DD.trace())

print(simplify(DD.det()))
