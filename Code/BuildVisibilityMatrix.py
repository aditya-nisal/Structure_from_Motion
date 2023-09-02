from scipy.sparse import lil_matrix
import numpy as np
def BuildVisibilityMatrix(n_camreas, n_points, cam_indices, point_indices):
    r = cam_indices.size * 2
    c = n_camreas * 7 + n_points * 3
    V = lil_matrix((r, c), dtype=int)

    i = np.arange(cam_indices.size)

    for s in range (7):
        V[2 * i, cam_indices * 7 + s] = 1
        V[2 * i + 1, cam_indices * 7 + s] = 1

    for s in range (3):
        V[2 * i, cam_indices * 3 + s] = 1
        V[2 * i + 1, cam_indices * 3 + s] = 1

    return V    