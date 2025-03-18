import math
import pyqtgraph as pg
import numpy as np
from collections import deque
from tqdm import tqdm
from matplotlib import pyplot as plt

# Suppress scientific notation in NumPy arrays
np.set_printoptions(suppress=True)
import warnings
import numpy as np

# Convert RuntimeWarnings to exceptions
warnings.simplefilter("error", RuntimeWarning)


def C2PoincareDisk(C):
    if C.ndim == 2:
        x_, y_ = (C[0, 1] / C[1, 1], np.sqrt(np.linalg.det(C)) / C[1, 1])
    else:
        dets = np.linalg.det(C)
        dets = np.clip(dets, 0, np.inf)
        x_, y_ = (C[:, 0, 1] / C[:, 1, 1], np.sqrt(dets) / C[:, 1, 1])

    x = (x_**2 + y_**2 - 1) / (x_**2 + (y_ + 1) ** 2)
    y = 2 * x_ / (x_**2 + (y_ + 1) ** 2)

    return x, y


def generate_matrix(v1, v2):
    # Discontinuous yielding of pristine micro-crystals - page 8/207
    # F = v1x v2x
    #     v1y v2y
    # C = F^TF

    F = np.array([v1, v2]).transpose()

    # F *= 1
    C = F.transpose() @ F
    C *= 1

    return F, C


def F2C(F):
    # C = F^TF
    C = np.einsum("ijk,jlk->ilk", np.transpose(F, (1, 0, 2)), F)
    return C


def C2F(C):
    """
    We are free to choose the rotation of F, therefore we set the first
    vector to have a y-component of 0.

                     a  b           a^2 ab
    If we assume F = 0  d, then C = ab  b^2+d^2
    We now only have to choose between a negative and positive sign, and we
    choose positive
    """

    if C.ndim == 2:
        a = np.sqrt(C[0, 0])
        b = C[0, 1] / a
        d = np.sqrt(C[1, 1] - b * b)
        return np.array([[a, b], [0, d]])

    elif C.ndim == 3:
        a = np.sqrt(C[:, 0, 0])
        b = C[:, 0, 1] / a
        d = np.sqrt(C[:, 1, 1] - b * b)
        zero = np.zeros_like(a)
        return np.array([[a, b], [zero, d]]).transpose(2, 0, 1)


def C2V(C):
    F = C2F(C)
    return F[:, 0], F[:, 1]


def constrainDeterminant(fixed, adjust):
    # Decrease the length of the adjust vector such that the determinant is fixed
    F, C = generate_matrix(fixed.head.pos(), adjust.head.pos())

    det = np.linalg.det(C)
    adjust.head.setPos(adjust.head.pos() / np.sqrt((abs(det))))  # remove abs?


def old_lagrange_reduction(v1, v2):
    # A Variational Model for Reconstructive Phase Transformations in Crystals
    # Page 75/7
    printing = False
    changing = True

    """
    This is still very much speculation, but in terms of moving with shears, 
    there are four simple shears one can do, in a way, moving on a 2d plane. 
                                                1 -1
    We will say that the familiar shear m3 = 0  1 is called 'left', and 
    consequently, m3^-1 will be right. When transposing these shears, we find
    'up' and 'down' also. 
    
    The reason we call m3 left, is that if we assume v1 = (1,0) and v2=(0,1),
    then Fm3 will have the effect of subtracting v1 from v2, resulting in 
    v2=(-1,1), ie. moving to the left.
    
                            1 -1
    left    =   m3      = 0  1
    
                            1  1
    right   =   m3^-1   = 0  1
    
                            1  0
    down    =   m3^T    =-1  1
    
                            1  0
    up      =   m3^-1^T = 1  1
    
    When we start the lagrange reduction process, we will say that we start
    in the 'left' direction (-1,0), and if we perform a reduction (v2=v1-v2),
    we will say that we move to the left (pos += (-1,0)). If we ever swap
    v1 and v2, this means that we have transposed m3, so we transpose the 
    direction we are moving in (-1,0) -> (0,-1). If the dot product is
    negative, we flipp the sign of the direction we are moving in 
    (-1,0) -> (1,0). 
    
    With this, we get a x,y integer possition for each 'region' of the 
    lagrange transformation thing.
    """

    direction = np.array([-1, 0])
    possition = np.array([0, 0])
    m1 = 0
    m2 = 0
    m3 = 0

    while changing:
        changing = False

        if v1.dot(v2) < 0:
            print("Dot product is negative") if printing else None
            v2 *= -1
            changing = True
            direction *= -1
            m1 += 1

        if v1.length() > v2.length():
            print("v1 is longer") if printing else None
            temp = v1
            v1 = v2
            v2 = temp
            changing = True
            direction = np.flip(direction, 0)
            m2 += 1

        if (v1 - v2).length() < v2.length():
            print("Difference is longer than v2") if printing else None
            v2 = v1 - v2
            changing = True
            possition += direction
            m3 += 1

    print("d ", direction) if printing else None
    print("p ", possition) if printing else None
    return v1, v2, m1, m2, m3


def lagrange_reduction(v1, v2):
    F, C_R = generate_matrix(v1, v2)  # Note that C_ is not reduced yet.

    m1 = 0
    m2 = 0
    m3 = 0
    max_m = 50
    ms = []
    m = np.identity(2)
    changed = True
    while changed:
        changed = False

        if C_R[0, 1] < 0:
            flip(C_R, 0, 1)
            lag_m1(m)
            changed = True
            m1 += 1
            if len(ms) < max_m:
                ms.append(1)

        if C_R[1, 1] < C_R[0, 0]:
            swap(C_R, 0, 0, 1, 1)
            lag_m2(m)
            changed = True
            m2 += 1
            if len(ms) < max_m:
                ms.append(2)

        if 2 * C_R[0, 1] > C_R[0, 0]:
            C_R[1, 1] += C_R[0, 0] - 2 * C_R[0, 1]
            C_R[0, 1] -= C_R[0, 0]
            lag_m3(m)
            changed = True
            m3 += 1
            if len(ms) < max_m:
                ms.append(3)

    C_R[1, 0] = C_R[0, 1]

    C_E_R = C_R.copy()

    if m1 % 2 == 1:
        flip(C_E_R, 0, 1)
    if m2 % 2 == 1:
        swap(C_E_R, 0, 0, 1, 1)
    C_E_R[1, 0] = C_E_R[0, 1]

    v1, v2 = C2V(C_R)
    return v1, v2, C_R, C_E_R, m, m1, m2, m3, ms


def flip(matrix, row, col):
    matrix[row, col] *= -1


def swap(matrix, row1, col1, row2, col2):
    temp = matrix[row1, col1]
    matrix[row1, col1] = matrix[row2, col2]
    matrix[row2, col2] = temp


def lag_m1(matrix):
    flip(matrix, 0, 1)
    flip(matrix, 1, 1)


def lag_m2(matrix):
    swap_cols(matrix)


# applies m3 n times
def lag_m3(matrix, n=1):
    # https://www.wolframalpha.com/input?i=%7B%7B1%2C+-1%7D%2C+%7B0%2C+1%7D%7D%5En
    multiplier_matrix = np.array([[1, -n], [0, 1]])

    new_matrix = matrix @ multiplier_matrix
    np.copyto(matrix, new_matrix)


def swap_cols(matrix):
    matrix[:, [0, 1]] = matrix[:, [1, 0]]


def fast_lagrange_reduction(v1, v2):
    """
    The idea here is that sometimes when one vector is very small, an the other
    one is very big (and/or the angle is close to 0 or 180), the lagrange reduction
    algorithm is very ineficcient. I have noticed a pattern, and i think i can
    reproduce the result much more quickly in these scenarios. As an example,
    one common pattern is this m transformation.
    m1 m2 m3 m3 m3 ... m3 m3 m1
    ei, this is the sequence in which you would need to transform your m matrix
    in the lagrange reduction process.
    """

    F, C_ = generate_matrix(v1, v2)
    # this is not a specific criteria. Should be further looked into TODO
    if v1.length() < 1 or v2.length() < 1:
        m1 = m2 = m3 = 0
        m = np.identity(2)

        if C_[0, 1] < 0:
            flip(C_, 0, 1)
            lag_m1(m)
            m1 += 1

        if C_[1, 1] < C_[0, 0]:
            swap(C_, 0, 0, 1, 1)
            lag_m2(m)
            m2 += 1

        a = C_[0, 0]
        b = C_[0, 1]
        d = C_[1, 1]
        # Now we find out how many m3 steps we need using
        # https://www.wolframalpha.com/input?i=+%7B%7B1%2C0%7D%2C%7B-1%2C1%7D%7D%5En+%7B%7Ba%2Cb%7D%2C%7Bb%2Cd%7D%7D%7B%7B1%2C-1%7D%2C%7B0%2C1%7D%7D%5En+        # We find what N is required so that 2C_[0,1] = C_[0,0]. Then we can
        # round up to ensure that  2b < a.
        # From hand calculation, I think that N=b/a-1/2
        N = b / a - 1 / 2
        N = int(np.ceil(N))
        C_[1, 1] = -N * (b - a * N) - b * N + d
        C_[0, 1] = b - N * a
        # we apply lag_m3 N times
        lag_m3(m, N)
        m3 = N

        # And now we check if that made b negative
        if C_[0, 1] < 0:
            flip(C_, 0, 1)
            lag_m1(m)
            m1 += 1

        # we remember to make C_ symetric
        C_[1, 0] = C_[0, 1]
        v1, v2 = C2V(C_)

    else:
        v1, v2, C_, m, m1, m2, m3 = lagrange_reduction(v1, v2)
    return v1, v2, C_, m, m1, m2, m3


def slow_lagrange_reduction_visualization(
    width, height, ppu, v2_is_fixed=False, loops=2
):
    # There are lots of strange things going on here...

    pixel_width, pixel_height = width * ppu, height * ppu
    x, y = np.meshgrid(
        np.linspace(-width, width, pixel_width),
        np.linspace(-height, height, pixel_height),
    )
    if v2_is_fixed:
        v1x = np.copy(x)
        v1y = np.copy(y)
        # v2 is fixed to always be (0,1) (ignoring determinant preservation adjustments)
        v2x = np.zeros_like(x)
        # we now want to scale
        C = ((1, v1y), (v1y, v1x**2 + v1y**2))
        det = v1x**2
        v2y = 1 / np.sqrt(det)
    else:
        v2x = np.copy(x)
        v2y = np.copy(y)
        # V1 is fixed to always be (1,0) (ignoring determinant preservation adjustments)
        v1y = np.zeros_like(x)
        # we now want to scale
        C = ((1, v2x), (v2x, v2x**2 + v2y**2))
        det = v2y**2
        v1x = 1 / np.sqrt(det)

    mState = np.zeros((*v1x.shape, 3), dtype=int)
    total_iterations = x.size  # or np.prod(x.shape)

    # Use tqdm with the total number of iterations
    for _ in range(3):
        for i, j in tqdm(np.ndindex(x.shape), total=total_iterations):
            v1, v2, C_, m, m1, m2, m3 = lagrange_reduction(
                (v1x[i, j], v1y[i, j]), (v2x[i, j], v2y[i, j])
            )
            v1x[i, j] = v1[0]
            v1y[i, j] = v1[1]
            v2x[i, j] = v2[0]
            v2y[i, j] = v2[1]
            mState[i, j] += [m1, m2, m3]

    colors = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 0, 0],
        ]
    )

    colorIndex = mState[:, :, 0] + 2 * mState[:, :, 1] + mState[:, :, 2]

    colorIndex = np.clip(colorIndex, 0, len(colors) - 1)
    colorMap = colors[colorIndex]

    np.clip(mState, 0, 256)
    heatmap = pg.ImageItem(colorMap, autoDownsample=False)
    return heatmap


def lagrange_reduction_visualization(width, height, ppu, v2_is_fixed=False, loops=2):
    # There are lots of strange things going on here...

    pixel_width, pixel_height = width * ppu, height * ppu
    x, y = np.meshgrid(
        np.linspace(-width, width, pixel_width),
        np.linspace(-height, height, pixel_height),
    )
    if v2_is_fixed:
        v1x = np.copy(x)
        v1y = np.copy(y)
        # v2 is fixed to always be (0,1)
        v2x = np.zeros_like(x)
        # we want to scale the second vector to preserve the determinant
        # C = ((1, v1y),
        #     (v1y, v1x**2+v1y**2))
        det = v1x**2
        v2y = 1 / np.sqrt(det)

        F = np.array(
            [
                [v1x, v2x],  # F has v1x and v2x as its first row,
                [v1y, v2y],
            ]
        )  # and v1y and v2y as its second row.
    else:
        v2x = np.copy(x)
        v2y = np.copy(y)
        # V1 is fixed to always be (1,0)
        v1y = np.zeros_like(x)
        # we now want to scale
        # C = ((1, v2x),
        #     (v2x, v2x**2+v2y**2))
        det = v2y**2
        v1x = 1 / np.sqrt(det)

        F = np.array(
            [
                [v2x, v1x],  # F has v1x and v2x as its first row,
                [v2y, v1y],
            ]
        )  # and v1y and v2y as its second row.

    # Now compute C = F^TF
    C = np.einsum("ijmn,jkmn->ikmn", np.transpose(F, (1, 0, 2, 3)), F)

    # We keep trac of the m state of each area pixel and convert this to color
    # later
    mState = np.zeros((*v1x.shape, 3), dtype=int)

    # Indexes to track state changes
    m1, m2, m3 = 0, 1, 2

    for i in tqdm(range(loops)):
        # If this is confusing see Note 2 at the bottom of the document
        # Create masks
        # Negate v2 where mask1 is True
        mask1 = C[0, 1] < 0
        C[0, 1, mask1] *= -1
        mState[mask1, m1] += 1

        mask2 = C[1, 1] < C[0, 0]
        # Swap operation
        C[0, 0, mask2], C[1, 1, mask2] = C[1, 1, mask2].copy(), C[0, 0, mask2].copy()
        mState[mask2, m2] += 1

        mask3 = 2 * C[0, 1] > C[0, 0]
        C[1, 1, mask3] += C[0, 0, mask3] - 2 * C[0, 1, mask3]
        C[0, 1, mask3] -= C[0, 0, mask3]
        mState[mask3, m3] += 1

    # Function to get a color palette from a colormap with 'num_colors' number of colors
    def get_color_palette(cmap_name, num_colors):
        import matplotlib.cm as cm

        cmap = cm.get_cmap(cmap_name)
        colors = cmap(np.linspace(0, 1, num_colors))[:, :3]  # Exclude the alpha channel
        return colors

    # Example color maps to try
    color_maps = [
        "viridis",
        "plasma",
        "inferno",
        "magma",
        "cividis",
        "Pastel1",
        "tab10",
    ]

    colors = get_color_palette(color_maps[4], 8)

    # on the left side (negative x), m1 will always be 1 higher than "normal"
    # so we remove it
    mState[: int(len(mState) / 2), :, m1] -= 1
    colorIndex = mState[:, :, m2]

    colorIndex = np.clip(colorIndex, 0, len(colors) - 1)
    colorMap = (
        colors[colorIndex]
        - np.array([1, 1, 1])[np.newaxis, np.newaxis, :]
        * 0.1
        * mState[:, :, m3][:, :, np.newaxis]
        - np.array([1, 1, 1])[np.newaxis, np.newaxis, :]
        * 0.1
        * mState[:, :, m1][:, :, np.newaxis]
    )
    # mask = np.all(mState == [2,1,1], axis=-1)
    # print(sum(sum(mask)))
    # Now use the mask to index into the colors array
    # colorMap = colors[mask.astype(int)]

    # The image is flipped for some reason
    colorMap = np.transpose(colorMap, (1, 0, 2))
    np.clip(mState, 0, 256)
    heatmap = pg.ImageItem(colorMap, autoDownsample=False)
    return heatmap


def angleBetweenPoints(p1, p2):
    # Calculate angle for the arrow at the end of the line
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.degrees(math.atan2(dy, dx))


def CToAngle(C):
    xy = C2PoincareDisk(C)
    start = xy[0]
    stop = xy[-1]

    # We want to find the angle between the origin and start,
    # and between the origin and stop

    origin = [0, 0]
    start_angle = angleBetweenPoints(origin, start)
    stop_angle = angleBetweenPoints(origin, stop)

    return (round(start_angle), round(stop_angle))


def euclidianDistance(x, y):
    return np.sqrt(x**2 + y**2)


def manhattanDistance(x, y):
    return abs(x) + abs(y)


def chebyshevDistance(x, y):
    return max([abs(x), abs(y)])


def generate_flood_fill_coordinates(max_distance, distanceFunction):
    visited = set()
    queue = deque([(0, 0)])
    visited.add((0, 0))
    coords = []

    while queue:
        x, y = queue.popleft()
        coords.append((x, y))

        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if distanceFunction(nx, ny) <= max_distance and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append((nx, ny))
    return coords


# Calculate C at a specific 'possition'
# @lru_cache(maxsize=None)
def CPos(C, x, y):
    """

    C can be transformed in 4 directions, similar to moving on a 2D plane:

        up      (+y):   m^T C m
        down    (-y):   m^T^-1 C m^-1
        right   (+x):   m C m^T
        left    (-x):   m^-1 C m^T^-1

    In order to move to 'possition' (1, -1), we would do

        m^T^-1 m C m^T m^-1

    Since this calculation can be done sequentially, we can save a lot of
    computation by always starting from (0,0), and recursively moving one
    step at a time. For example, if we want to calculate (-3,0), (-2,0) and
    (-1,0), instead of doing the full calculation for all of the possitions,
    we can use the solution from a neighboring point. lru_cache take care of
    all of this by saving the solutions for a given set of arguments to the
    function.
    """

    nr = len(C)
    one = np.array([1] * nr)
    zero = np.array([0] * nr)

    m3 = np.array([[one, -1 * one], [zero, one]]).transpose(2, 0, 1)

    m3Inv = np.linalg.inv(m3)

    def up(C):
        return conTrans(C, m3)

    def down(C):
        return conTrans(C, m3Inv)

    def right(C):
        return conTrans(C, m3.transpose(0, 2, 1))

    def left(C):
        return conTrans(C, m3Inv.transpose(0, 2, 1))

    if x == 0:
        if y == 0:
            return C
        elif y < 0:
            return CPos(up(C), x, y + 1)
        elif y > 0:
            return CPos(down(C), x, y - 1)
    elif x < 0:
        return CPos(right(C), x + 1, y)
    elif x > 0:
        return CPos(left(C), x - 1, y)


# Congruence Transform
def conTrans(A, m):
    return m.transpose(0, 2, 1) @ A @ m


"""
Extra comments that are a bit too long to put in the relevant code



Note 1:
    # (See Homogeneous nucleation of dislocations as a pattern formation phenomenon)
    
    When drawing in the configuration space, here are some guidelines:
    All values must satisfy det=1 and C12=C21.
    
    When your equation is drawn in the configuration space, you can mirror
    the line through the y-axis by swaping v1 and v2. In other words, 
    sequentually swaping the columns and rows of C, or algebraically,
    using 
    
    m2 = ((0,1),
            1,0))
    
    we can express
    
    C_y_mirrored = m2^T C m2 (Note that m2^T=m2)
    
    such that 
    
    C = ((a,b),  -->  C_y_mirrored = ((d,c),
            (c,d))                       (b,a))
    
    
    Similarly you can mirror the line through the x-axis by subtracting pi/2
    from the (smallest) angle between v1 and v2, or flipping
    the sign of both components of either v1 or v2. In terms of C, this is 
    equivalent to flipping the sign of C12 and C21.
    
    This can be done algebraically using 
    m1 = ((1, 0),
            (0,-1))
        in a similar fassion as shown above.
        
    C_x_mirrored = m1^T C m1 (Note that m1^T=m1)
    C = ((a,b),  -->  C_x_mirrored = (( d,-c),
            (c,d))                       (-b, a))

    The third transformation, m3


Note 2:

    np arrays can use arrays of bools of the same shape to mask a selection.
    In practice, this means that if you have three objects: x = [A, B, C],
    and a bool mask: maks = [False, True, True], you can make a selection 
    of your objects by doing x[mask]. 
    
    In our case, v1x, v1y, v2x and v2y are NxM matrixes, and combining them into
        mask1 = v1x**2 + v1y**2 > v2x**2 + v2y**2
    we get a mask is true in every spot that v1 is longer than v2. This would
    look something like this:
    
    '-' is False
    'O' is True
    
    This is where v2 (the vertical (0,1) vector) is variable and v1 is fixed at (1,0).
    Keep also in mind that as v2 moves around, v1 is adjusted to preserve the determinant.
    (Run the program and try moving the vertical vector around)

        - - - - - - - - - - - - - - - - - - - -
        - - - - - - - - - - - - - - - - - - - -
        - - - - - - - - - - - - - - - - - - - -
        - - - - - - - - - - - - - - - - - - - -
        - - - - - - - - - - - - - - - - - - - -
        - - - - - - - - O O O O - - - - - - - -
        - - - - - O O O O O O O O O O - - - - -
        - O O O O O O O O O O O O O O O O O O -
        O O O O O O O O O O O O O O O O O O O O
        O O O O O O O O O O O O O O O O O O O O
        O O O O O O O O O O O O O O O O O O O O
        O O O O O O O O O O O O O O O O O O O O
        - O O O O O O O O O O O O O O O O O O -
        - - - - - O O O O O O O O O O - - - - -
        - - - - - - - - O O O O - - - - - - - -
        - - - - - - - - - - - - - - - - - - - -
        - - - - - - - - - - - - - - - - - - - -
        - - - - - - - - - - - - - - - - - - - -
        - - - - - - - - - - - - - - - - - - - -
        - - - - - - - - - - - - - - - - - - - -


    This is where v1 is variable, but because of coloring reasons, we have swapped the vectors,
    so this is true when v2 is longer than v1.

        - - - - - - - - O O O O - - - - - - - -
        - - - - - - - O O O O O O - - - - - - -
        - - - - - - - O O O O O O - - - - - - -
        - - - - - - - O O O O O O - - - - - - -
        - - - - - - - O O O O O O - - - - - - -
        - - - - - - O O O O O O O O - - - - - -
        - - - - - - O O O O O O O O - - - - - -
        - - - - - - O O O O O O O O - - - - - -
        - - - - - O O O O O O O O O O - - - - -
        - - - - - O O O O O O O O O O - - - - -
        - - - - - O O O O O O O O O O - - - - -
        - - - - - O O O O O O O O O O - - - - -
        - - - - - - O O O O O O O O - - - - - -
        - - - - - - O O O O O O O O - - - - - -
        - - - - - - O O O O O O O O - - - - - -
        - - - - - - - O O O O O O - - - - - - -
        - - - - - - - O O O O O O - - - - - - -
        - - - - - - - O O O O O O - - - - - - -
        - - - - - - - O O O O O O - - - - - - -
        - - - - - - - - O O O O - - - - - - - -


"""
