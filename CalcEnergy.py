import numpy as np
from Fields import DDfield
from Utilities import dot

muB = 5.7883818012e-2

def energies(spins, nbl, J, K, Z, DD, pos, DMI, muS):
    E_x = 0.0
    E_dm = 0.0
    if J != 0.0:
        for j in range(spins.shape[0]):
            for k in range(len(nbl[j])):
                E_x -= J * dot(spins[j], spins[nbl[j][k]])
    if np.any(K):
        E_an = - np.sum(K * spins[:,2] * spins[:,2])
    else:
        E_an = 0.0
    if np.any(np.asarray(Z)):
        E_z = -np.sum(spins * (Z * (muB * muS)))
    else:
        E_z = 0.0
    if DD:
        Hdd = DDfield(pos, spins, muS)
        E_dd = -0.5 * np.sum(spins * (Hdd * (muB * muS)))
    else:
        E_dd = 0.0
    if np.any(DMI):
        for j in range(spins.shape[0]):
            for k in range(len(nbl[j])):
                E_dm -= np.dot(DMI[j][k], np.cross(spins[j], spins[k]))
    E = E_x + E_an + E_z + E_dd + E_dm
    return E

# =============================================================================
# def exEn(spins, nbl, J, muS):
#     Hx = xfield(spins, nbl, J, muS)
#     E_x = -0.5 * np.sum(spins * Hx)
#     return E_x
# =============================================================================

def main():
    from DipoleTensor import DipoleTensor
    
    spins = np.array([[0.1, -0.1, 0.989949], [0.1, -0.1, 0.989949]])
    muS = 1.0
    nbl = []
    J = 0.0
    K = 0.0
    Z = 0.0
    DD = True
    pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
    rij = pos[0] - pos[1]
    Dmat = DipoleTensor(rij)
    E = energies(spins, nbl, J, K, Z, DD, pos, muS)
    
if __name__=="__main__":
    main()
