import numpy as np

def LinearTriangulation(M_1, C_2, R_2, K, inliers):

    points_1 = inliers[:, 0:2]
    points_2 = inliers[:, 2:4]

    ones = np.ones((points_1.shape[0], 1))
    points_1 = np.hstack((points_1, ones))
    points_2 = np.hstack((points_2, ones))

    I = np.identity(3)
    M_2 = np.hstack((I, -C_2))
    M_2 = np.dot(K, np.dot(R_2, M_2))

    X_list = []

    for p_1, p_2 in zip(points_1, points_2):
        A = [p_1[0]*M_1[2, :] - M_1[0, :]]
        A.append(p_1[1]*M_1[2, :] - M_1[1, :])
        A.append(p_2[0]*M_2[2, :] - M_2[0, :])
        A.append(p_2[1]*M_2[2, :] - M_2[1, :])

        _, __, V_t = np.linalg.svd(np.array(A))

        V = V_t.T
        X = V[:, -1]

        X = X / X[3]
        X = X[:3]

        X = np.array(X)
        X = X.reshape((3, 1))
        X_list.append(X)
        
    return np.array(X_list)