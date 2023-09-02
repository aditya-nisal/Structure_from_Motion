import numpy as np

def LinearPnP(img_2d_3d, K):
    A = np.empty((0, 12), np.float32)
    for img_pt in img_2d_3d:
        x, y = img_pt[0], img_pt[1]
        normalised_pts = np.dot(np.linalg.inv(K), np.array([[x], [y], [1]]))
        normalised_pts = normalised_pts/normalised_pts[2]

        X = img_pt[2:]
        X = X.reshape((3, 1))

        X = np.append(X, 1)

        Z = np.zeros((4, ))
        A_1 = np.hstack((Z, -X.T, normalised_pts[1]*(X.T)))
        A_2 = np.hstack((X.T, Z, -normalised_pts[0]*(X.T)))
        A_3 = np.hstack((-normalised_pts[1]*(X.T), normalised_pts[0]*X.T, Z))

        for a in [A_1, A_2, A_3]:
            A = np.append(A, [a], axis = 0)

    A = np.float32(A)
    _, __, V_t = np.linalg.svd(A)

    V = V_t.T

    pose = (V[:, -1]).reshape((3, 4))

    R_ = (pose[:, 0:3]).reshape((3, 3))
    T_ = (pose[:, 3]).reshape((3, 1))

    U, __, V_t = np.linalg.svd(R_)
    R_ = np.dot(U, V_t)

    if np.linalg.det(R_) < 0 :
        R_ = -R_
        T_ = -T_

    C_ = -np.dot(R_, T_)
    return R_, C_

    