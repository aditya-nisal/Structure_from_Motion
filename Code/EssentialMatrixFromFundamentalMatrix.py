import numpy as np

def EssentialMatrixFromFundamentalMatrix(F, K):

    E = np.dot(np.dot(K.T, F), K)

    U, S, V_t = np.linalg.svd(E)

    S = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])

    E = np.dot(np.dot(U, S), V_t)

    return E