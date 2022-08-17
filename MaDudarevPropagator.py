import numpy as np
from Utilities import dot, cross
from copy import copy

muB = 5.7883818012e-2
gamma = 0.1760859644

def Evolve(spins, Z, dt, eta, muS):
    sp = copy(spins)
    H = np.linalg.norm(Z)
    xi = -(gamma / (1 + eta**2)) * H * dt
    zeta = -H * eta * (gamma / (1 + eta**2)) * dt
    for i in range(sp.shape[0]):
        chi = dot(sp[i], Z) / H
        den = H * (1 + np.exp(2 * zeta) + chi * (1 - np.exp(2 * zeta)))
        a1 = 2 * H * np.exp(zeta) * np.cos(xi)
        a2 = 2 * np.exp(zeta) * np.sin(xi)
        a3 = (1 - np.exp(2 * zeta) + chi * (1 + np.exp(2 * zeta) - 
                                            2 * np.exp(2 * zeta) * 
                                            np.cos(xi))) * 0.5 #* muS * muB 
        sp[i] = (a1 * sp[i] + a2 * cross(sp[i], Z) + Z * a3) / den
    return sp

def main():
    spin = np.array([[1.0, 0.0, 0.0]])
    muS = 1
    Z = np.array([0.0, 0.0, 10.0])
    eta = 0.1
    dt = 0.001
    n = 50000
    
    k = 501
    md = int(n/(k-1))
    
    mag = np.zeros(shape = (k, 3))
    mag[0,0] = spin[0][0]
    mag[0,1] = spin[0][1]
    mag[0,2] = spin[0][2]
    
    cnt = 1
    for i in range(n):
        spin = Evolve(spin, Z, dt, eta, muS)
        if (i+1)%md==0:
            mag[cnt,0] = spin[0][0]
            mag[cnt,1] = spin[0][1]
            mag[cnt,2] = spin[0][2]
            cnt += 1
            
    dt = -dt
    mag2 = np.zeros(shape = (k, 3))
    mag2[-1,0] = spin[0][0]
    mag2[-1,1] = spin[0][1]
    mag2[-1,2] = spin[0][2]
    cnt = 1
    for i in range(n):
        spin = Evolve(spin, Z, dt, eta, muS)
        if (i+1)%md==0:
            mag2[k-1-cnt, 0] = spin[0][0]
            mag2[k-1-cnt, 1] = spin[0][1]
            mag2[k-1-cnt, 2] = spin[0][2]
            cnt += 1
            
    np.save("MaDudarevForw", mag)
    np.save("MaDudarevBack", mag2)
    

if __name__=="__main__":
    main()