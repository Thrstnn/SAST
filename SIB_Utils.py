import numpy as np

# Determinant of matrix 
# | x1 x2 x3 |
# | y1 y2 y3 |
# | z1 z2 z3 |
def det3(x1, x2, x3, y1, y2, y3, z1, z2, z3):
    d = x1*(y2*z3 - y3*z2) - x2*(y1*z3 - y3*z1) + x3*(y1*z2 - y2*z1)
    return d

# Solution to the linear equation Ax = b with Cramer's rule
def solveLinEq(spin, a, dt):
    A = np.array([[1, -a[2] * dt * 0.5, a[1] * dt * 0.5], 
                  [a[2] * dt * 0.5, 1, -a[0] * dt * 0.5], 
                  [-a[1] * dt * 0.5, a[0] * dt * 0.5, 1]])
    A2 = np.transpose(A)
    b = np.dot(A2, spin)
    
    detA = det3(A[0][0], A[0][1], A[0][2], 
                A[1][0], A[1][1], A[1][2], 
                A[2][0], A[2][1], A[2][2])
    
    detX = det3(b[0], A[0][1], A[0][2], 
                b[1], A[1][1], A[1][2], 
                b[2], A[2][1], A[2][2])
    
    detY = det3(A[0][0], b[0], A[0][2], 
                A[1][0], b[1], A[1][2], 
                A[2][0], b[2], A[2][2])
    
    detZ = det3(A[0][0], A[0][1], b[0], 
                A[1][0], A[1][1], b[1], 
                A[2][0], A[2][1], b[2])
    
    x = detX / detA
    y = detY / detA
    z = detZ / detA
    return np.array([x, y, z])