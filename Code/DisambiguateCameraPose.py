import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
from LinearTriangulation import LinearTriangulation
from Misc.utils import PlotFuncs

def cheirality_check(C, R, X_list):
    count = 0
    r_3 = R[2]

    for X in X_list:
        if(np.dot(r_3, X - C) > 0):
            count+=1
    return count

def DisambiguateCameraPose(M_1, C_2_list, R_2_list, K, inliers):
    count_max = 0
    i = 0
    fg = plt.figure()
    fx = fg.add_subplot(111)
    p_f = PlotFuncs()

    print("Plotting the 4 camera poses ans their points \n")

    for R, C in zip(R_2_list, C_2_list):
        C = C.reshape((3, 1))
        X_list = LinearTriangulation(M_1, C, R, K, inliers)
        count = cheirality_check(C, R, X_list)

        p_f.plot_triangulated_points(X_list, i, C, R)
        print(count)

        if(count > count_max):
            count_max = count
            R_best = R
            C_best = C
            X_list_best = X_list
            index = i

        i+=1
    
    plt.xlim(-15, 20)
    plt.ylim(-30, 40)
    plt.show()

    fig = plt.figure()
    fx = fig.add_subplot(111)
    p_f.plot_triangulated_points(X_list_best, index, C_best, R_best)
    plt.xlim(-15, 20)
    plt.ylim(-30, 40)
    plt.show()

    print("\n")
    return R_best, C_best, X_list_best, index