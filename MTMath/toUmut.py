import numpy as np

def conTrans(C, M):
    """Apply a congruence transform: return M^T @ C @ M.

    Works for C with shape (2, 2) or (N, 2, 2).
    M can be (2, 2) for a single transform broadcast over all slices, or (N, 2, 2)
    to use a different matrix per slice (matching the leading dimension of C).
    """
    C_arr = np.asarray(C)
    M_arr = np.asarray(M, dtype=C_arr.dtype)

    return np.swapaxes(M_arr, -1, -2) @ C_arr @ M_arr

def transformC(C, transformation, inverse=False):
    if transformation is None:
        return C
    elif isinstance(transformation, np.ndarray):
        # Use the provided matrix directly as a congruence transform
        # (broadcasts over C if C has a leading dimension)
        M = transformation
    elif transformation.lower() == "none":
        return C
    elif transformation == "triangular":
        gamma = (4 / 3) ** (1 / 4)
        # pre_M = gamma*np.array([[-1.0, 0.0], [0.5, -np.sqrt(3) / 2]])
        pre_M = gamma * np.array([[1.0, 0.5], [0.0, np.sqrt(3) / 2]])
        pre_M = np.linalg.inv(pre_M)
        # Optional: go the other direction
        # pre_M = -pre_M
        M = pre_M
    else:
        raise ValueError(f"Unknown transformation: {transformation}")

    if inverse:
        Minv = np.linalg.inv(M)
        return conTrans(C, Minv)
    else:
        return conTrans(C, M)
    
def C2PoincareDisk(C, transformation=None, eps=1e-12):
    """
    Map a symmetric 2x2 matrix C to (x, y) on the Poincaré disk by:
      (i)  normalizing C so det(C)=1 (if det>0),
      (ii) projecting the normalized matrix to (x,y).

    Supports a single 2x2 or a batch of shape (..., 2, 2).
    Returns x, y
    """
    C = np.asarray(C)
    C = transformC(C, transformation)

    a = C[..., 0, 0]
    b = C[..., 0, 1]
    c = C[..., 1, 1]
    # Determinant and validity
    # Dont use np.linalg.det. It is less numerically stable
    det = a * c - b * b
    valid = det > eps

    # Scale to det=1 without taking sqrt on invalid entries
    scale = np.empty_like(det, dtype=float)
    scale[valid] = np.sqrt(det[valid])
    scale[~valid] = np.nan
    C_hat = C / scale[..., None, None]  # det(C_hat) = 1 where valid

    # Projection to (x,y) from the det=1 surface (stereographic-style inverse)
    c11 = C_hat[..., 0, 0]
    c12 = C_hat[..., 0, 1]
    c22 = C_hat[..., 1, 1]

    t = 1.0 / (2.0 + c11 + c22)
    x = t * (c11 - c22)
    y = 2 * t * c12

    return x, y


def poincareDisk2C(X, Y, transformation=None, eps=1e-12):
    r = 1.0 - X**2 - Y**2
    if np.any(r < 0):
        raise ValueError("Point outside of disk!")
    safe_r = np.where(np.abs(r) <= eps, np.nan, r)
    t = 2.0 / safe_r
    C11 = t * (1.0 + X) - 1.0
    C22 = t * (1.0 - X) - 1.0
    C12 = t * Y

    C = np.stack(
        [
            np.stack([C11, C12], axis=-1),  #
            np.stack([C12, C22], axis=-1),
        ],
        axis=-2,
    )

    C = transformC(C, transformation, inverse=True)
    return C

def generate_poincare_disk(
    resolution=500,
    zoom=1,
    returnMask=False,
    transformation=None,
    eps=1e-9,
):
    # Define the range for x and y based on the unit circle
    radius = 1.0 / zoom

    x_min, x_max = -radius, radius
    y_min, y_max = -radius, radius

    # Create the meshgrid for the x and y coordinates
    X, Y = np.meshgrid(
        np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution)
    )
    # Calculate the mask for points inside the unit circle
    # (We don't need to use radius or zoom here because its only to avoid infinities anyway)
    mask = (X**2 + Y**2) >= (1 - eps)
    X[mask] = np.nan
    Y[mask] = np.nan

    C = poincareDisk2C(X, Y, transformation=transformation)

    if returnMask:
        return C, mask
    return C

if __name__ == "__main__":
    C = generate_poincare_disk(100)
    # C_e1 = Do elastic reduction 1
    # C_e2 = Do elastic reduction 2
    # Check that C_e1 == C_e2 (except nan values)