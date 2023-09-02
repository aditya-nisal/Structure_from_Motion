import numpy as np
from scipy import optimize

def ReprojectionError(points_img1, M_1, points_img2, M_2, X):
    X = np.reshape(X, (3, 1))
    X = np.vstack((X, 1))    

    rep_error = 0

    point_proj_img1 = np.dot(M_1,X)
    point_proj_img1 = point_proj_img1/point_proj_img1[2]

    point_proj_img2 = np.dot(M_2, X)
    point_proj_img2 = point_proj_img2/point_proj_img2[2]

    rep_error += ((points_img1[0] - point_proj_img1[0])**2) + ((points_img1[1] - point_proj_img1)**2)

    rep_error += ((points_img2[0] - point_proj_img2[0])**2) + ((points_img2[1] - point_proj_img2)**2)
    return rep_error

def optimize_params(x0, points_img1, M_1, points_img2, M_2):

    reproj_err = ReprojectionError(points_img1, M_1, points_img2, M_2, x0)
    # print("len of tupe = {}".format(len(reproj_err)))
    return np.ravel(reproj_err)


def NonlinearTriangulation(M_1, M_2, X_list, inliers, K):
    points_img1 = inliers[:, 0:2]
    points_img2 = inliers[:, 2:4]

    X_list_ref = []

    for point_img1, point_img2, X in  zip(points_img1, points_img2, X_list):
        X = X.reshape(X.shape[0],)
        # print(len(X))
        res = optimize.least_squares(fun=optimize_params, x0=X, args = [point_img1, M_1, point_img2, M_2])
        x_ref = res.x
        x_ref = np.reshape(X, (3,))
        X_list_ref.append(x_ref)
    X_list_ref = np.array(X_list_ref)
    X_list_ref = X_list_ref.reshape((X_list_ref.shape[0], 3))
    
    return X_list_ref