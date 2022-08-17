#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 13:13:43 2022

@author: thorsteinn
"""

import numpy as np
from copy import copy
from SIB_Utils import solveLinEq

gamma = 0.1760859644
muB = 5.7883818012e-2
kB = 8.617333262145e-2

def FindField(spins, dt, nbl, J, K, ek, Z, eta, DD, pos, muS, DMI):
    Ddip = 0.9274009994 * muS
    H = np.zeros_like(spins)
    for i in range(spins.shape[0]):
        wx, wy, wz = 0.0, 0.0, 0.0
        
        # Exchange
        for j in range(len(nbl[i])):
            wx += 2 * J * spins[nbl[i][j]][0] / (muB * muS)
            wy += 2 * J * spins[nbl[i][j]][1] / (muB * muS)
            wz += 2 * J * spins[nbl[i][j]][2] / (muB * muS)
        # Zeeman
        wx += Z[0]
        wy += Z[1]
        wz += Z[2]
        
        # Dipole
        if DD == True:
            for j in range(spins.shape[0]):
                if j != i:
                    lenr = np.linalg.norm(pos[i] - pos[j])
                    rij = (pos[i] - pos[j]) / lenr
                    EffField = (Ddip / lenr**3) * (3 * np.dot(spins[j], rij) * 
                                                   rij - spins[j])
                    wx += EffField[0]
                    wy += EffField[1]
                    wz += EffField[2]
            
        # Dzyaloshinskii-Moriya
        for j in range(len(nbl[i])):
            DM = (np.cross(spins[nbl[i][j]], DMI[i][nbl[i][j]]) + 
                  np.cross(DMI[nbl[i][j]][i], spins[nbl[i][j]])) / (muS * muB)
            wx += DM[0]
            wy += DM[1]
            wz += DM[2]
            
        # Anisotropy                
        # TODO: Allow any anisotropy direction
        # TODO: Allow multiple anisotropy axes
        H_an = (2 * K[i] / (muB * muS)) * np.dot(spins[i], ek[i]) * ek[i]
        wx += H_an[0]
        wy += H_an[1]
        wz += H_an[2]
        
        H[i][0] = wx
        H[i][1] = wy
        H[i][2] = wz
    return H


def SIB(spins, dt, nbl, J, K, ek, Z, eta, DD, pos, muS, DMI):
    Pred = np.zeros_like(spins)
    pre = gamma / (1 + eta * eta)
    H = FindField(spins, dt, nbl, J, K, ek, Z, eta, DD, pos, muS, DMI)
    for i in range(spins.shape[0]):
        a1 = np.zeros(3)
        a1[0] = -H[i][0] * pre - eta * pre * (spins[i][1] * H[i][2] - spins[i][2] * H[i][1])
        a1[1] = -H[i][1] * pre - eta * pre * (spins[i][2] * H[i][0] - spins[i][0] * H[i][2])
        a1[2] = -H[i][2] * pre - eta * pre * (spins[i][0] * H[i][1] - spins[i][1] * H[i][0])
        Pred[i] = solveLinEq(spins[i], a1, dt)
    
    sp2 = (Pred + spins) * 0.5
    H2 = FindField(sp2, dt, nbl, J, K, ek, Z, eta, DD, pos, muS, DMI)
    for i in range(spins.shape[0]):
        a2 = np.zeros(3)
        a2[0] = -H2[i][0] * pre - eta * pre * (sp2[i][1] * H2[i][2] - sp2[i][2] * H2[i][1])
        a2[1] = -H2[i][1] * pre - eta * pre * (sp2[i][2] * H2[i][0] - sp2[i][0] * H2[i][2])
        a2[2] = -H2[i][2] * pre - eta * pre * (sp2[i][0] * H2[i][1] - sp2[i][1] * H2[i][0])
        spins[i] = solveLinEq(spins[i], a2, dt)
    
def main():
    from BuildNbl import Constructnbl
    import matplotlib.pyplot as plt
    from CalcEnergy import energies
    from time import time
    
# =============================================================================
#     sz = (1, 1, 1)
#     sp = np.array([[1.0, 0.0, 0.0]])
#     pos = np.array([[0.0, 0.0, 0.0]])
# =============================================================================
    
# =============================================================================
#     sz = (1, 1, 2)
#     sp = np.array([[0.1, -0.1, 0.989949366], [0.4, 0.2, 0.894427191]])
#     pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
# =============================================================================
    
    sz = (1, 1, 50)
    sp = np.array([[0.1, -0.1, 0.989949366], 
                   [0.4, 0.2, 0.894427191],
                   [1.0, 0.0, 0.0],
                   [0.4, -0.2, 0.894427191],
                   [0.0, 0.0, 1.0],
                   [-0.4, 0.2, 0.894427191],
                   [0.1, -0.1, -0.989949366],
                   [0.0, -1.0, 0.0],
                   [-0.1, 0.1, 0.989949366],
                   [-0.4, 0.2, -0.894427191],
                   [0.1, -0.1, 0.989949366], 
                   [0.4, 0.2, 0.894427191],
                   [1.0, 0.0, 0.0],
                   [0.4, -0.2, 0.894427191],
                   [0.0, 0.0, 1.0],
                   [-0.4, 0.2, 0.894427191],
                   [0.1, -0.1, -0.989949366],
                   [0.0, -1.0, 0.0],
                   [-0.1, 0.1, 0.989949366],
                   [-0.4, 0.2, -0.894427191],
                   [0.1, -0.1, 0.989949366], 
                   [0.4, 0.2, 0.894427191],
                   [1.0, 0.0, 0.0],
                   [0.4, -0.2, 0.894427191],
                   [0.0, 0.0, 1.0],
                   [-0.4, 0.2, 0.894427191],
                   [0.1, -0.1, -0.989949366],
                   [0.0, -1.0, 0.0],
                   [-0.1, 0.1, 0.989949366],
                   [-0.4, 0.2, -0.894427191],
                   [0.1, -0.1, 0.989949366], 
                   [0.4, 0.2, 0.894427191],
                   [1.0, 0.0, 0.0],
                   [0.4, -0.2, 0.894427191],
                   [0.0, 0.0, 1.0],
                   [-0.4, 0.2, 0.894427191],
                   [0.1, -0.1, -0.989949366],
                   [0.0, -1.0, 0.0],
                   [-0.1, 0.1, 0.989949366],
                   [-0.4, 0.2, -0.894427191],
                   [0.1, -0.1, 0.989949366], 
                   [0.4, 0.2, 0.894427191],
                   [1.0, 0.0, 0.0],
                   [0.4, -0.2, 0.894427191],
                   [0.0, 0.0, 1.0],
                   [-0.4, 0.2, 0.894427191],
                   [0.1, -0.1, -0.989949366],
                   [0.0, -1.0, 0.0],
                   [-0.1, 0.1, 0.989949366],
                   [-0.4, 0.2, -0.894427191],])
    pos = np.array([[0.0, 0.0, 0.0], 
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 2.0],
                    [0.0, 0.0, 3.0],
                    [0.0, 0.0, 4.0],
                    [0.0, 0.0, 5.0],
                    [0.0, 0.0, 6.0],
                    [0.0, 0.0, 7.0],
                    [0.0, 0.0, 8.0],
                    [0.0, 0.0, 9.0],
                    [0.0, 0.0, 10.0], 
                    [0.0, 0.0, 11.0],
                    [0.0, 0.0, 12.0],
                    [0.0, 0.0, 13.0],
                    [0.0, 0.0, 14.0],
                    [0.0, 0.0, 15.0],
                    [0.0, 0.0, 16.0],
                    [0.0, 0.0, 17.0],
                    [0.0, 0.0, 18.0],
                    [0.0, 0.0, 19.0],
                    [0.0, 0.0, 20.0], 
                    [0.0, 0.0, 21.0],
                    [0.0, 0.0, 22.0],
                    [0.0, 0.0, 23.0],
                    [0.0, 0.0, 24.0],
                    [0.0, 0.0, 25.0],
                    [0.0, 0.0, 26.0],
                    [0.0, 0.0, 27.0],
                    [0.0, 0.0, 28.0],
                    [0.0, 0.0, 29.0],
                    [0.0, 0.0, 30.0], 
                    [0.0, 0.0, 31.0],
                    [0.0, 0.0, 32.0],
                    [0.0, 0.0, 33.0],
                    [0.0, 0.0, 34.0],
                    [0.0, 0.0, 35.0],
                    [0.0, 0.0, 36.0],
                    [0.0, 0.0, 37.0],
                    [0.0, 0.0, 38.0],
                    [0.0, 0.0, 39.0],
                    [0.0, 0.0, 40.0], 
                    [0.0, 0.0, 41.0],
                    [0.0, 0.0, 42.0],
                    [0.0, 0.0, 43.0],
                    [0.0, 0.0, 44.0],
                    [0.0, 0.0, 45.0],
                    [0.0, 0.0, 46.0],
                    [0.0, 0.0, 47.0],
                    [0.0, 0.0, 48.0],
                    [0.0, 0.0, 49.0],])
    
    n_s = sz[0] * sz[1] * sz[2]
    
    J = 1.0
    K = np.ones(sz[2]) * 1.0
    ek = np.zeros(shape = (sp.shape[0], 3))
    for i in range(ek.shape[0]):
        ek[i] = np.array([0.0, 0.0, 1.0])
        ek[i] = ek[i] / np.linalg.norm(ek[i])
    DD = False
    Z = np.array([0.0, 0.0, 0.5])
    DMI = np.zeros(shape = (n_s,n_s,3))
# =============================================================================
#     DMI[0][1] = np.array([1.0, 1.0, 1.0])
#     DMI[1][0] = np.array([-1.0, -1.0, -1.0])
# =============================================================================
    
    muS = 1.0
    T = 0.0
    eta = 0.1
    nbl = Constructnbl(sz, pbc = [False, False, False])
    
    dt = 0.001
    n = 60000
    
# =============================================================================
#     En = np.zeros(n+1)
#     E = energies(sp, nbl, J, K, Z, DD, pos, DMI, muS)
#     En[0] = E / n_s
# =============================================================================
    
# =============================================================================
#     magF1 = np.zeros(shape = (n+1,3))
#     magF1[0] = sp[0]
# =============================================================================
# =============================================================================
#     magF2 = np.zeros(shape = (n+1,3))
#     magF2[0] = sp[1]
# =============================================================================
    start = time()
    for i in range(n):
        SIB(sp, dt, nbl, J, K, ek, Z, eta, DD, pos, muS, DMI)
#        magF1[i+1] = sp[0]
#        magF2[i+1] = sp[1]
    end = time()
    el = end - start
    print(f"Elapsed time SIB Damped 50 spins : {el}")
        #E = energies(sp, nbl, J, K, Z, DD, pos, DMI, muS)
        #En[i+1] = E / n_s
#    dt = -dt
# =============================================================================
#     En2 = np.zeros(n+1)
#     E = energies(sp, nbl, J, K, Z, DD, pos, DMI, muS)
#     En2[-1] = E / (sz[0] * sz[1] * sz[2])
# =============================================================================
    
# =============================================================================
#     magB1 = np.zeros(shape = (n+1,3))
#     magB1[-1] = sp[0]
#     magB2 = np.zeros(shape = (n+1,3))
#     magB2[-1] = sp[1]
#     for i in range(n):
#         SIB(sp, dt, nbl, J, K, ek, Z, eta, DD, pos, muS, DMI)
#         #E = energies(sp, nbl, J, K, Z, DD, pos, DMI, muS)
#         #En2[n-i-1] = E / (sz[0] * sz[1] * sz[2])
#         magB1[n-i-1] = sp[0]
#         magB2[n-i-1] = sp[1]
#     mag1 = magF1 - magB1
#     mag2 = magF2 - magB2
#     magF = np.concatenate((magF1, magF2), axis = 1)
#     magB = np.concatenate((magB1, magB2), axis = 1)
#    # EnFB = np.column_stack((En, En2))
#      
#     mg = np.concatenate((mag1, mag2), axis = 1)
# =============================================================================
   # EnDiff = En - En2
   # np.savetxt(r"ComparisonTests/ComponentDiffSIB_DissCorr.txt", mg)
   # np.savetxt(r"ComparisonTests/EnergyDiffSIB_SingleSpin5.txt", EnDiff)
   # np.savetxt(r"ComparisonTests/TimestepSIB_1.txt", magF1)
   # np.savetxt(r"ComparisonTests/ComponentsSIBBack_CorrectedOneSpin1.txt", magB)
   # np.savetxt(r"ComparisonTests/EnergySIB_SingleSpin5.txt", EnFB)
    
# =============================================================================
#     t = np.linspace(0, n * np.abs(dt), n+1)
#     plt.xlabel("Time [ps]")
#     plt.ylabel("Energy [meV]")
#     plt.plot(t[::1], En[::1])
# =============================================================================

if __name__=='__main__':
    main()