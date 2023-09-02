import numpy as np

def CheckDeterminant(C, R):
    if np.linalg.det(R) < 0:
        return -C, -R
    
    else: 
        return C, R

def ExtractCameraPose(E):
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    U, _, V_t = np.linalg.svd(E)

    C_1 = U[:, 2]
    R_1 = (np.dot(U, np.dot(W, V_t)))
    C_2 = -U[:, 2]
    R_2 = (np.dot(U, np.dot(W, V_t)))
    C_3 = U[:, 2]
    R_3 = (np.dot(U, np.dot(W.T, V_t)))
    C_4 = -U[:, 2]
    R_4 = (np.dot(U, np.dot(W.T, V_t)))


    C_1, R_1 = CheckDeterminant(C_1, R_1)
    C_2, R_2 = CheckDeterminant(C_2, R_2)
    C_3, R_3 = CheckDeterminant(C_3, R_3)
    C_4, R_4 = CheckDeterminant(C_4, R_4)


    C = np.array([C_1, C_2, C_3, C_4])
    R = np.array([R_1, R_2, R_3, R_4])

    return C, R