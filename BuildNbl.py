# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 13:36:52 2022

@author: torst
"""
import numpy as np

# =============================================================================
# Construct the neighbourlist of the system. Assumes SC system.
# The argument size should be given as a tuple containing 3 integers, e.g. 
# size = (3,3,1). pbc can be given as a boolean for the whole system or as a 
# list for periodicity in only specific directions, e.g. pbc = [True, True, False].
# Returns a dictionary with a key-value pair containing the index of a site as
# the key and a list of its nearest neighbours as the value.
# =============================================================================


def Constructnbl(size, pbc):
    if isinstance(pbc, bool):
        pbcx = pbc
        pbcy = pbc
        pbcz = pbc
    else:
        pbcx = pbc[0]
        pbcy = pbc[1]
        pbcz = pbc[2]
    nbl = {}
    for k in range(size[2]):
        for j in range(size[1]):
            for i in range(size[0]):
                l = []
                if pbcx:
                    l.append((i+1)%size[0] + j*size[0] + k*size[0]*size[1])
                    l.append((i-1)%size[0] + j*size[0] + k*size[0]*size[1])
                else:
                    if size[0] < 2:
                        pass
                    elif i == (size[0]-1):
                        l.append((i-1) + j*size[0] + k*size[0]*size[1])
                    elif i == 0:
                        l.append((i+1) + j*size[0] + k*size[0]*size[1])
                    else:
                        l.append((i-1) + j*size[0] + k*size[0]*size[1])
                        l.append((i+1) + j*size[0] + k*size[0]*size[1])
                if pbcy:
                    l.append(i + ((j+1)%size[1])*size[0] + k*size[0]*size[1])
                    l.append(i + ((j-1)%size[1])*size[0] + k*size[0]*size[1])
                else:
                    if size[1] < 2:
                        pass
                    elif j == (size[1]-1):
                        l.append(i + (j-1)*size[0] + k*size[0]*size[1])
                    elif j == 0:
                        l.append(i + (j+1)*size[0] + k*size[0]*size[1])
                    else:
                        l.append(i + (j+1)*size[0] + k*size[0]*size[1])
                        l.append(i + (j-1)*size[0] + k*size[0]*size[1])
                if pbcz:
                    l.append(i + j*size[0] + ((k+1)%size[2])*size[0]*size[1])
                    l.append(i + j*size[0] + ((k-1)%size[2])*size[0]*size[1])
                else:
                    if size[2] < 2:
                        pass
                    elif k == (size[2]-1):
                        l.append(i + j*size[0] + (k-1)*size[0]*size[1])
                    elif k == 0:
                        l.append(i + j*size[0] + (k+1)*size[0]*size[1])
                    else:
                        l.append(i + j*size[0] + (k-1)*size[0]*size[1])
                        l.append(i + j*size[0] + (k+1)*size[0]*size[1])
                nbl[i+j*size[0]+k*size[0]*size[1]] = l
    return nbl
    
def main():
    size = (1,1,1)
    pbc = [False, False, False]
    nbl = Constructnbl(size, pbc)
    for i in nbl:
        print(i, nbl[i])
        
if __name__ == '__main__':
    main()
         