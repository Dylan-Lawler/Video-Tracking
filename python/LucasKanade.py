import numpy as np
from scipy.interpolate import RectBivariateSpline

def create_interpolator(image):
    """
    Creates a bivariate spline interpolator for the given image.
    """
    return RectBivariateSpline(np.arange(image.shape[0]), np.arange(image.shape[1]), image)

def compute_warped_coordinates(X, Y, p):
    """
    Computes the coordinates warped by the movement vector p.
    """
    X_warp = X + p[0]
    Y_warp = Y + p[1]
    return X_warp, Y_warp

def compute_derivatives(I, Y_warp, X_warp):
    """
    Computes the derivatives of the image at the warped coordinates.
    """
    Ix = I.ev(Y_warp, X_warp, dx=0, dy=1).flatten()  # derivative in x direction
    Iy = I.ev(Y_warp, X_warp, dx=1, dy=0).flatten()  # derivative in y direction
    return Ix, Iy

def compute_error_image(T, I, Y, X, Y_warp, X_warp):
    """
    Computes the difference between the template and the current image at the warped coordinates.
    """
    return T.ev(Y.flatten(), X.flatten()) - I.ev(Y_warp.flatten(), X_warp.flatten())

def LucasKanade(It, It1, rect):
    threshold = 0.01875
    maxIters = 100
    p = np.zeros(2)
    x1, y1, x2, y2 = rect

    I = create_interpolator(It1)
    T = create_interpolator(It)
    X, Y = np.meshgrid(np.arange(x1, x2), np.arange(y1, y2))
    i = 0
    threshpassed = False
    # put your implementation here
    while i < maxIters and threshpassed == False:
        X_warp, Y_warp = compute_warped_coordinates(X, Y, p)
        Ix, Iy = compute_derivatives(I, Y_warp, X_warp)
        J = np.vstack((Ix, Iy)).T
        b = compute_error_image(T, I, Y, X, Y_warp, X_warp)
        dp, _, _, _ = np.linalg.lstsq(J, b, rcond=None)
        p += dp.flatten()  # Ensure dp is correctly added to p
        
        if  np.linalg.norm(dp) < threshold: 
            threshpassed = True

    return p
