import numpy as np
from scipy.interpolate import RectBivariateSpline

import numpy as np
from scipy.interpolate import RectBivariateSpline

def create_interpolator(image):
    """
    Creates a bivariate spline interpolator for the given image.
    """
    return RectBivariateSpline(np.arange(image.shape[0]), np.arange(image.shape[1]), image)

def warp_coordinates(x, y, p):
    """
    Warps coordinates using the provided affine parameters.
    """
    x_warp = x*(1+p[0]) + y*p[1] + p[2]
    y_warp = x*p[3] + y*(1+p[4]) + p[5]
    return x_warp, y_warp

def compute_jacobian(x, y, Ix, Iy):
    """
    Computes the Jacobian matrix for the affine warp.
    """
    return np.vstack((x*Ix, y*Ix, Ix, x*Iy, y*Iy, Iy)).T

def update_parameters(J, b):
    """
    Solves for dp, the update to the parameters, in the least squares sense.
    """
    return np.linalg.lstsq(J, b, rcond=None)[0]

def LucasKanadeAffine(It, It1, rect):
    threshold = 0.01875
    maxIters = 100
    p = np.zeros(6)
    x1, y1, x2, y2 = map(int, [max(0, rect[0]), max(0, rect[1]), min(It.shape[1], rect[2]), min(It.shape[0], rect[3])])

    I = create_interpolator(It1)
    T = create_interpolator(It)

    X, Y = np.meshgrid(np.arange(x1, x2), np.arange(y1, y2))
    Tx = T.ev(Y, X)

    i = 0 
    threshpassed = False
    while i < maxIters and threshpassed == False:
        x_warp, y_warp = warp_coordinates(X, Y, p)
        Ix = I.ev(y_warp, x_warp, dy=1).flatten()
        Iy = I.ev(y_warp, x_warp, dx=1).flatten()

        J = compute_jacobian(X.flatten(), Y.flatten(), Ix, Iy)
        b = (Tx - I.ev(y_warp, x_warp)).flatten()

        dp = update_parameters(J, b)
        p += dp
        
        if  np.linalg.norm(dp) < threshold: 
            threshpassed = True
        
        i += 1

    M = np.array([[1.0 + p[0], p[1], p[2]], [p[3], 1.0 + p[4], p[5]]])
    return M

