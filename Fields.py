# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 14:25:59 2022

@author: torst
"""

#Assumption: All spins have length 1. Factor mu0 omitted.

import numpy as np
from BuildNbl import Constructnbl

# =============================================================================
# The exchange field. Calculated using nearest neighbours.
# =============================================================================

def xfield(spins, nbl, J, muS):
    H = np.zeros_like(spins)
    muB = 5.7883818012e-2
    for i in range(H.shape[0]):
        l = nbl[i]
        for j in l:
            H[i] += 2 * J * spins[j] / (muB * muS)
    return H

# =============================================================================
# The magnetocrystalline anisotropy field. Can calculate both for local
# anisotropy or a global value.
# =============================================================================

def anfield(spins, K, e_k = np.array([0.0, 0.0, 1.0])):
    from Utilities import dot
    if isinstance(K, float):
        K = np.ones(spins.shape[0]) * K
    K = np.asarray(K)
    H = np.zeros_like(spins)
    for i in range(spins.shape[0]):
        H[i] = 2*K[i]*dot(spins[i], e_k) * e_k
    return H

# =============================================================================
# The exchange field at a single site. Calculated using nearest neighbours.
# =============================================================================

def xfield2(spins, nbl, J, ind):
    H = np.zeros(3)
    l = nbl[ind]
    for j in l:
        H += J * spins[j]
    return H

# =============================================================================
# The magnetocrystalline anisotropy field at a single site. Can calculate both for local
# anisotropy or a global value.
# =============================================================================

def anfield2(spins, K, ind, e_k = np.array([0.0, 0.0, 1.0])):
    from Utilities import dot
    if isinstance(K, float):
        K = np.ones(spins.shape[0]) * K
    K = np.asarray(K)
    H = np.zeros(3)
    H = +2*K[ind]*dot(spins[ind], e_k) * e_k
    return H
    
# =============================================================================
# Dipole field. Brute force
# =============================================================================
def DDfield(pos, spins, muS):
    from Utilities import dot
    from copy import copy
    from DipoleTensor import DipoleTensor
    Ddip = 0.9274009994 * muS
    H = np.zeros_like(spins)
    for i in range(spins.shape[0]):
        Hx, Hy, Hz = 0.0, 0.0, 0.0
        for j in range(spins.shape[0]):
                if j != i:
                    lenr = np.linalg.norm(pos[i] - pos[j])
                    rij = (pos[i] - pos[j]) / lenr
                    EffField = (Ddip / lenr**3) * (3 * dot(spins[j], rij) * rij - spins[j])
                    Hx += EffField[0]
                    Hy += EffField[1]
                    Hz += EffField[2]
        H[i] = np.array([Hx, Hy, Hz])
    return H

def main():
    pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    spins = np.array([[1.0, 0.0, 0.0], [0.70710678, 0.70710678, 0.0]])
    D = DDfield(pos, spins)
    
if __name__=="__main__":
    main()