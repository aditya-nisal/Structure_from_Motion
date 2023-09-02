# import numpy as np
# import random
# np.random.seed(2)

# from EstimateFundamentalMatrix import EstimateFundamentalMatrix

# def GetInliersRansac(text_points, threshold = 0.005):
#     img1_points = text_points[:, 0:2]
#     img2_points = text_points[:, 2:4]
#     ones = np.ones((img1_points.shape[0], 1))
#     img1_points = np.hstack((img1_points, ones))
#     img2_points = np.hstack((img2_points, ones))
#     random.seed(42)
#     max_inliers = 0


#     for i in range(10000):
        
#         points = np.array(random.sample(text_points, 8), np.float32)
#         F = EstimateFundamentalMatrix(points)

#         values = np.abs(np.diag(np.dot(np.dot(img2_points, F), img1_points.T)))

#         index_inliers = np.where(values<threshold)
#         index_outliers = np.where(values>=threshold)
        
#         if np.shape(index_inliers[0])[0] > max_inliers:
#             max_inliers = np.shape(index_inliers[0])[0]
#             index_max_inliers = index_inliers
#             index_min_outliers = index_outliers
#             F_max_inliers = F
    
#     F_max_inliers = EstimateFundamentalMatrix(text_points[index_max_inliers])

#     print("max inliers: {}", (max_inliers))

#     return text_points[index_max_inliers], text_points[index_min_outliers], F_max_inliers, text_points[:, 0:2], text_points[:, 2:4]

import numpy as np
np.random.seed(2)

from EstimateFundamentalMatrix import EstimateFundamentalMatrix

def GetInliersRansac(text_points, threshold = 0.005):
    img1_points = text_points[:, 0:2]
    img2_points = text_points[:, 2:4]
    ones = np.ones((img1_points.shape[0], 1))
    img1_points = np.hstack((img1_points, ones))
    img2_points = np.hstack((img2_points, ones))
    np.random.seed(42)
    max_inliers = 0

    for i in range(10000):
        points = text_points[np.random.choice(text_points.shape[0], 8, replace=False)]
        F = EstimateFundamentalMatrix(points)

        values = np.abs(np.diag(np.dot(np.dot(img2_points, F), img1_points.T)))

        index_inliers = np.where(values<threshold)
        index_outliers = np.where(values>=threshold)
        
        if np.shape(index_inliers[0])[0] > max_inliers:
            max_inliers = np.shape(index_inliers[0])[0]
            index_max_inliers = index_inliers
            index_min_outliers = index_outliers
            F_max_inliers = F
    
    F_max_inliers = EstimateFundamentalMatrix(text_points[index_max_inliers])

    print("max inliers: {}", (max_inliers))

    return text_points[index_max_inliers], text_points[index_min_outliers], F_max_inliers, text_points[:, 0:2], text_points[:, 2:4]
