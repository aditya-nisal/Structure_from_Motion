import numpy as np
import random
from LinearPNP import LinearPnP
def AllReprojetionError(img_points, M, X_all, ret = False):
    o = np.ones((X_all.shape[0], 1))
    # print("o.shape = {}".format(o.shape))
    # print("X_all.shape = {}".format(X_all.shape))
    X_all = np.hstack((X_all, o))
    # print("Works fine?")
    X_all = X_all.T
    
    proj_points = (np.dot(M, X_all))
    proj_points = proj_points.T
    proj_points[:, 0] = proj_points[:, 0]/proj_points[:, 2]
    proj_points[:, 1] = proj_points[:, 1]/proj_points[:, 2]
    proj_points = proj_points[:, 0:2]

    reproj_error = (img_points - proj_points)**2
    reproj_error = np.sum(reproj_error, axis=1)

    if(ret):
        return reproj_error, proj_points
    return reproj_error


def PnPRANSAC(correspond_2d_3d, K, thresh = 20):
    max_inliers = 0

    for i in range(10000):
        corr6 = correspond_2d_3d[np.random.choice(correspond_2d_3d.shape[0], 6, replace = False)]
        R, C = LinearPnP(corr6, K)
        C = C.reshape((3, 1))
        I = np.identity(3)
        M = np.hstack((I, -C))
        M = np.dot(K, np.dot(R, M))

        img_points = correspond_2d_3d[:, 0:2]
        X_all = correspond_2d_3d[:, 2:]

        reproj_error = AllReprojetionError(img_points, M, X_all)
        loc = np.where(reproj_error < thresh)[0]
        count = np.shape(loc)[0]
        if count > max_inliers:
            max_inliers = count
            inliers = correspond_2d_3d[loc]
            R_ = R
            C_ = C

    best_pose = np.hstack((R_, C_))
    print("Max inliers: ")
    print(max_inliers)

    return best_pose, inliers