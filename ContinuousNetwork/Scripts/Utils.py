import numpy as np

#It extracts the entries correspondent to boundary conditions

def getBCs(trajectories):
    bcs = {"q1":trajectories[:,:2],
       "q2":trajectories[:,-4:-2],
       "v1":trajectories[:,2:4],
       "v2":trajectories[:,-2:]}
    return bcs