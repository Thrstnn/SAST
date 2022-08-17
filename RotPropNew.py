import numpy as np
from MaDudAnglesEvolve import Evolve
from Utilities import Cart2Sph, Sph2Cart, RotateCoorSys
from copy import copy
from CalcEnergy import energies

gamma = 0.1760859644    # rad / (ps * T)
muB = 5.7883818012e-2   # meV / T
kB = 8.617333262145e-2  # meV / K

def FindField(i, spins, future_S, dt, nbl, J, K, ek, Z, eta, 
              DD, pos, muS, Anis, DMI):
    Ddip = 0.9274009994 * muS # mu0*muB/(4*pi*(1e-10)**3) * muS
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
    if Anis == False:
        H = np.array([wx, wy, wz])
        FutSAppr = spins[i] + ((gamma / (1 + eta**2)) * np.cross(H, spins[i]) - \
            gamma * eta / (1 + eta**2) * np.cross(spins[i], np.cross(spins[i], H))) * dt
        H_an = (K[i] / (muB * muS)) * np.dot(ek, spins[i] + FutSAppr) * ek 
        wx += H_an[0]
        wy += H_an[1]
        wz += H_an[2]
    else:
        H_an = (K[i] / (muB * muS)) * np.dot(ek, spins[i] + future_S[i]) * ek
        wx += H_an[0]
        wy += H_an[1]
        wz += H_an[2]
    
    H_vec = np.array([wx, wy, wz])
    H = np.linalg.norm(H_vec)
    H_vec = H_vec / H
    return H, H_vec

def RotateAndEvolve(i, sp, future_S, dt, nbl, J, K, ek, Z, eta, DD, pos,
                    muS, Anis, DMI):
    H, H_vec = FindField(i, sp, future_S, dt, nbl, J, K, ek, Z, eta, 
                         DD, pos, muS, Anis, DMI)
    Q = RotateCoorSys(H_vec)
    u = Q[:,0]
    v = Q[:,1]
    w = Q[:,2]
    c1 = np.dot(u, sp[i])
    c2 = np.dot(v, sp[i])
    c3 = np.dot(w, sp[i])
    sp[i] = np.array([c1, c2, c3])
    r, theta, phi = Cart2Sph(sp[i][0], sp[i][1], sp[i][2])
    t, p = Evolve(theta, phi, H, dt, eta)
    x, y, z = Sph2Cart(r, t, p)
    sp[i] = np.array([x, y, z])
    sp[i] = np.dot(Q, sp[i])
    future_S[i] = sp[i]
    return

def ST2(spins, futureS1, futureS2, dt, nbl, J, K, ek, Z, eta, DD, pos, muS, DMI):
    sp = np.load("TmpSpins2.npy")
    n_sp = spins.shape[0]
    # Evolve spins in order by half timestep
    for i in range(n_sp-1):
        RotateAndEvolve(i, sp, futureS1, dt / 2, nbl, J, K, ek[i], Z, eta, DD, 
                             pos, muS, False, DMI)
    # Evolve the last spin by full timestep
    RotateAndEvolve(n_sp-1, sp, futureS1, dt, nbl, J, K, ek[-1], Z, eta, DD, 
                             pos, muS, False, DMI)
    # Evolve spins in reverse order by half timestep
    for i in range(n_sp-2, -1, -1):
        RotateAndEvolve(i, sp, futureS2, dt / 2, nbl, J, K, ek[i], Z, eta, DD, 
                             pos, muS, False, DMI)
    
    if np.any(K):
        E1 = np.inf
        E0 = energies(spins, nbl, J, K, Z, DD, pos, DMI, muS)
        while np.abs(E1-E0) > 1e-9:
            E1 = E0
            sp = np.load('TmpSpins2.npy')
            # Evolve spins in order by half timestep
            for i in range(n_sp-1):
                RotateAndEvolve(i, sp, futureS1, dt / 2, nbl, J, K, ek[i], Z, eta, DD, 
                                     pos, muS, True, DMI)
            # Evolve the last spin by full timestep
            RotateAndEvolve(n_sp-1, sp, futureS1, dt, nbl, J, K, ek[-1], Z, eta, DD, 
                                     pos, muS, True, DMI)
            # Evolve spins in reverse order by half timestep
            for i in range(n_sp-2, -1, -1):
                RotateAndEvolve(i, sp, futureS2, dt / 2, nbl, J, K, ek[i], Z, eta, DD, 
                                     pos, muS, True, DMI)
            E0 = energies(sp, nbl, J, K, Z, DD, pos, DMI, muS)
    return sp

def main():
    from BuildNbl import Constructnbl
    import matplotlib.pyplot as plt
    from time import time
    
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

    futureS1 = np.zeros(shape = (sp.shape[0], 3))
    futureS2 = np.zeros(shape = (sp.shape[0], 3))
    n_s = sz[0] * sz[1] * sz[2]
    
    J = 1.0
    K = np.ones(sz[2]) * 1.0
    ek = np.zeros(shape = (sp.shape[0], 3))
    for i in range(ek.shape[0]):
        ek[i] = np.array([0.0, 0.0, 1.0])
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
#     EnErr = np.zeros(n+1)
# =============================================================================
    
# =============================================================================
#     magF1 = np.zeros(shape = (n+1,3))
#     magF1[0] = sp[0]
#     magF2 = np.zeros(shape = (n+1,3))
#     magF2[0] = sp[1]
# =============================================================================
    
    start = time()
    for i in range(n):
        np.save("TmpSpins2", sp)
        sp = ST2(sp, futureS1, futureS2, dt, nbl, J, K, ek, Z, eta, DD, 
                 pos, muS, DMI)
# =============================================================================
#         E = energies(sp, nbl, J, K, Z, DD, pos, DMI, muS)
#         En[i+1] = E / n_s
#         EnErr[i+1] = En[0] - En[i+1]
# =============================================================================
# =============================================================================
#         magF1[i+1] = sp[0]
#         magF2[i+1] = sp[1]
# =============================================================================
    end = time()
    el = end - start
    print(f"Elapsed time ST Damped 50 spins: {el}")    
# =============================================================================
#     dt = -dt
#     En2 = np.zeros(n+1)
#     E = energies(sp, nbl, J, K, Z, DD, pos, DMI, muS)
#     En2[-1] = E / (sz[0] * sz[1] * sz[2])
#     
#     magB1 = np.zeros(shape = (n+1,3))
#     magB1[-1] = sp[0]
#     magB2 = np.zeros(shape = (n+1,3))
#     magB2[-1] = sp[1]
# # =============================================================================
# #     for i in range(n):
# #         np.save("TmpSpins2", sp)
# #         sp = ST2(sp, futureS1, futureS2, dt, nbl, J, K, ek, Z, eta, DD, 
# #                  pos, muS, DMI)
# #         E = energies(sp, nbl, J, K, Z, DD, pos, DMI, muS)
# #         En2[n-i-1] = E / (sz[0] * sz[1] * sz[2])
# #         magB1[n-i-1] = sp[0]
# #         magB2[n-i-1] = sp[1]
# # =============================================================================
#     
#     mag1 = magF1 - magB1
#     mag2 = magF2 - magB2
#     magF = np.concatenate((magF1, magF2), axis = 1)
#     magB = np.concatenate((magB1, magB2), axis = 1)
#     #EnFB = np.column_stack((En, En2))
#     
#     mg = np.concatenate((mag1, mag2), axis = 1)
# =============================================================================
    #EnDiff = En - En2
# =============================================================================
#     np.savetxt(r"ComparisonTests/ComponentDiffST_NoDiss.txt", mg)
#     np.savetxt(r"ComparisonTests/EnergyDiffST_NoDiss.txt", EnDiff)
#     np.savetxt(r"ComparisonTests/ComponentsSTForw_NoDiss.txt", magF)
#     np.savetxt(r"ComparisonTests/ComponentsSTBack_NoDiss.txt", magB)
#     np.savetxt(r"ComparisonTests/EnergyST_NoDiss.txt", EnFB)
# =============================================================================
        
# =============================================================================
#     t = np.linspace(0, n * np.abs(dt), n+1)
#     plt.xlabel("Time [ps]")
#     plt.ylabel("Energy [meV]")
#     plt.plot(t[::1], En[::1])
# =============================================================================

if __name__=="__main__":
    main()