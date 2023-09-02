from Misc.utils import MiscFuncs
import numpy as np
from scipy.optimize import least_squares

def ComputeProjError(M, image_point, X):

    X = X.reshape((3, 1))
    X = np.append(X, 1)

    project_impage_point = np.dot(M, X)
    project_impage_point[0] = project_impage_point[0]/project_impage_point[2]
    project_impage_point[1] = project_impage_point[1]/project_impage_point[2]

    reproj_error = ((image_point[0] - project_impage_point[0])**2) + ((image_point[1] - project_impage_point[1])**2)
    return reproj_error

def rot_to_quat(R):
	qxx,qyx,qzx,qxy,qyy,qzy,qxz,qyz,qzz = R.flatten()
	m = np.array([[qxx-qyy-qzz,0, 0, 0],[qyx+qxy,qyy-qxx-qzz,0,0],
		[qzx+qxz,qzy+qyz,qzz-qxx-qyy,0],[qyz-qzy,qzx-qxz,qxy-qyx,qxx+qyy+qzz]])/3.0
	val,vec = np.linalg.eigh(m)
	quat = vec[[3,0,1,2],np.argmax(val)]
	if quat[0]<0:
		quat = -quat

	return quat 

def quat_to_rot(quat):
	q_1,q_2,q_3,q_4 = quat
	Nq = q_1*q_1+q_2*q_2+q_3*q_3+q_4*q_4
	if Nq < np.finfo(np.float).eps:
		return np.eye(3)
	s = 2.0/Nq
	X = q_2*s
	Y = q_3*s
	Z = q_4*s
	wX = q_1*X; wY = q_1*Y; wZ = q_1*Z
	xX = q_2*X; xY = q_2*Y; xZ = q_2*Z
	yY = q_3*Y; yZ = q_3*Z; zZ = q_4*Z
	R =  np.array([[ 1.0-(yY+zZ), xY-wZ, xZ+wY ],
            [ xY+wZ, 1.0-(xX+zZ), yZ-wX ],
            [ xZ-wY, yZ+wX, 1.0-(xX+yY) ]])

	return R	


def optimize_params(x0, K, pts_img_all, X_all):

	# calculate reprojection error
	reproj_err_all = []
	R = quat_to_rot(x0[:4])
	C = x0[4:]

	misc_funcs = MiscFuncs()
	M = misc_funcs.get_projection_matrix(K, R, C)

	for pt_img, X in zip(pts_img_all, X_all):
		reproj_err = ComputeProjError(M, pt_img, X)
		reproj_err_all.append(reproj_err)
	
	reproj_err_all = np.array(reproj_err_all)
	reproj_err_all = reproj_err_all.reshape(reproj_err_all.shape[0],)

	return reproj_err_all


def NonLinearPnP(K, pose, corresp_2d_3d):
	poses_non_lin = {}
	corr = corresp_2d_3d
	img_points = corr[:, 0:2]
	X_all = corr[:, 2:]
	R = pose[:, 0:3]
	C = (pose[:, 3]).reshape((3, 1))
	Quat = rot_to_quat(R)
	x0 = np.append(Quat, C)
	res = least_squares(fun = optimize_params, x0 = x0, args = (K, img_points, X_all), ftol = 1e-10)
	O = res.x
	R_best = quat_to_rot(O[:4])
	C_best = (O[4:]).reshape((3, 1))
	pose_best = np.hstack((R_best, C_best))
	return pose_best
	
    