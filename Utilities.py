# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 10:18:13 2022

@author: torst
"""
import numpy as np

def dot(a, b):
    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)
    assert a.shape == (3,)
    assert b.shape == (3,)
    s = 0.0
    for i in range(3):
        s += a[i] * b[i]
    return s

def cross(a, b):
    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)
    assert a.shape == (3,)
    assert b.shape == (3,)
    c1 = a[1]*b[2]-a[2]*b[1]
    c2 = a[2]*b[0]-a[0]*b[2]
    c3 = a[0]*b[1]-a[1]*b[0]
    c = np.array([c1, c2, c3])
    return c

def Cart2Sph(x, y, z):
    h = np.array([0.0, 0.0, 1.0])
    a = np.array([x, y, z])
    la = np.linalg.norm(a)
    cr = np.linalg.norm(np.cross(h, a)) / la
    do = np.dot(h, a) / la
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(cr, do)
    phi = np.arctan2(y, x)
    return r, theta, phi

def Sph2Cart(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

# Rotate the coordinate system such that the effective field coincides with
# the z axis
def RotateCoorSys(H_vec):
    k = np.array([0.0, 0.0, 1.0])
    b = np.cross(k, H_vec)
    b = b / np.linalg.norm(b)
    
    theta = np.arctan2(np.linalg.norm(np.cross(H_vec, k)), np.dot(H_vec, k))
    if theta == 0.0:
        return np.identity(3)
    
    # Quaternion components
    q0 = np.cos(theta/2)
    q1 = np.sin(theta/2) * b[0]
    q2 = np.sin(theta/2) * b[1]
    q3 = np.sin(theta/2) * b[2]
    
    # Components of rotation matrix
    q00 = 1 - 2 * (q2**2 + q3**2)
    q01 = 2 * (q1*q2 - q0*q3)
    q02 = 2 * (q1*q3 + q0*q2)
    q10 = 2 * (q1*q2 + q0*q3)
    q11 = 1 - 2 * (q1**2 + q3**2)
    q12 = 2 * (q2*q3 - q0*q1)
    q20 = 2 * (q1*q3 - q0*q2)
    q21 = 2 * (q2*q3 + q0*q1)
    q22 = 1 - 2 * (q1**2 + q2**2)
    
    Q = np.array([[q00, q01, q02],[q10, q11, q12],[q20, q21, q22]])
    return Q