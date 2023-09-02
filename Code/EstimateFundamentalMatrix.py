import numpy as np

def EstimateFundamentalMatrix(pointss):
    A = []
    for points in pointss:

        x, y, x_dash, y_dash = points[0], points[1], points[2], points[3]

        A.append([x*x_dash, x*y_dash, x, y*x_dash, y*y_dash, y, x_dash, y_dash, 1])

    A = np.array(A)

    _, __, vt = np.linalg.svd(A)

    v = vt.T
    X = v[:, -1]

    F = np.reshape(X, (3, 3)).T
    U, S, V = np.linalg.svd(F)
    S[2 ]= 0.0
    S_1 = np.diag(S)
    F = U.dot(S_1).dot(V)

    return F