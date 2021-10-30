import numpy as np

# input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def binary_cross_entropy(Y, P):
    
    sum = 0 
    for i in range(len(Y)):
        sum += Y[i]*np.log(P[i]) + (1-Y[i])*np.log((1-P[i]))

    return -sum