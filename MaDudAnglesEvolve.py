import numpy as np

gamma = 0.1760859644

def Evolve(theta, phi, H, dt, eta):
    a = -(eta / (1+eta**2)) * gamma * H * dt
    b = (gamma / (1 + eta**2)) * H * dt
    t = 2 * np.arctan2(np.tan(theta / 2) * np.exp(a), 1)
    p = phi + b
    return t, p

def main():
    theta = np.pi / 2
    phi = 0
    H = 10
    dt = 0.001
    eta = 0.1
    n = 200000
    
# =============================================================================
#     k = 501
#     md = int(n/(k-1))
# =============================================================================
    
    ang = np.zeros(n+1)
    ang[0] = theta
    
    ph = np.zeros(n+1)
    ph[0] = phi
    
    for i in range(n):
        theta, phi = Evolve(theta, phi, H, dt, eta)
        ang[i+1] = theta
        ph[i+1] = phi
         
    dt = -dt
    ang2 = np.zeros(n+1)
    ang2[-1] = theta
    ph2 = np.zeros(n+1)
    ph2[-1] = phi
    for i in range(n):
        theta, phi = Evolve(theta, phi, H, dt, eta)
        ang2[n-1-i] = theta
        ph2[n-1-i] = phi
            
    np.save(r"ComparisonTests/MaDudAnglesForw", ang)
    np.save(r"ComparisonTests/MaDudAnglesBack", ang2)
    np.save(r"ComparisonTests/MaDudAzimuthForw", ph)
    np.save(r"ComparisonTests/MaDudAzimuthBack", ph2)
    
if __name__=="__main__":
    main()
    
    
    