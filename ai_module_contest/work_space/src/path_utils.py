import numpy as np
from astar.search import AStar
from scipy.interpolate import splprep, splev

def check_trav(p1, p2, kdtree):
    # check if line from p1 to p2 crosses a non-traversable area
    traversable = True
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    scalar = np.linspace(0, 1., 10)
    for k in scalar:
        new_pt = (p1[0] + k*dx, p1[1] + k*dy)
        dist, pt_idx = kdtree.query(new_pt)
        if (dist > 0.1):
            traversable = False
            break
    return traversable


def reduce_path(path):
    # remove unnecessary points from path
    new_path = [path[0]] # keep first point
    theta1 = 9999
    theta2 = 0

    for row in range(len(path)-1):
        pt1 = path[row]
        pt2 = path[row+1]

        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]

        theta2 = np.arctan2(dy, dx)
        dtheta = theta2 - theta1
        theta1 = theta2

        if (np.abs(dtheta) > 0.1):
            new_path.append(pt2)
        else:
            new_path = new_path[:-1]  # remove the last point
            new_path.append(pt2)
    return new_path


def clean_path(path, kdtree, point_matrix, xedges, yedges, counts):
    # use astar to manuever around obstacles
    new_path = []
    new_path.append(path[0])

    for row in range(len(path) - 1):
        # check that from p_n to p_(n+1) doesn't cross intraversable area and fix with astar if it does
        pt1 = path[row]
        pt2 = path[row+1]

        dist1, idx_pt1 = kdtree.query(pt1)
        dist2, idx_pt2 = kdtree.query(pt2)

        # if points not in traversable area replace with traversable points
        if dist1 > 0.015:
            pt1 = point_matrix[idx_pt1, :2]

        if dist2 > 0.015:
            pt2 = point_matrix[idx_pt2, :2]

        # does the path from current point to the next point cross a non-traversable area
        trav = check_trav(pt1, pt2, kdtree)

        if (trav == False):
            # if path crosses non-traversable region use A* to find a way around obstacles
            start_sqr = world2grid(xedges, yedges, pt1)
            end_sqr = world2grid(xedges, yedges, pt2)
            assert start_sqr != end_sqr, print("Error start == end")
            new_segment = AStar(counts.astype(np.int8).tolist()).search(start_sqr, end_sqr)
            if (new_segment is not None):
                new_segment = grid2world(xedges, yedges, new_segment)
                if (len(new_segment) > 5):
                    new_segment = smooth_path(new_segment, -1, s_var=8)  # smooth new segment; may cause crosses into intraversable territory
                new_path += new_segment
            else:
                new_segment = [pt1, pt2]
        else:
            new_path.append(pt2)
    return new_path


def smooth_path(path, n_samples, s_var=12):
    # use splines to smooth path
    path_order = np.array(path)
    tck, u = splprep([path_order[:, 0], path_order[:, 1]], s=s_var, k=5) # s sets tradeoff between smoothness and closeness of fit; higher numbers -> smoother; k sets curve order

    if n_samples == -1:
        per_samp = 0.2
        count = 0
        while (n_samples < 6)  and (count < 8):
            n_samples = int(path_order.shape[0] * per_samp)
            per_samp += 0.1
            count += 1
    u_new = np.linspace(u.min(), u.max(), n_samples)
    x_spline, y_spline = splev(u_new, tck)
    smoothed_path = list(zip(x_spline, y_spline))
    return smoothed_path


def world2grid(xedges, yedges, pos):
    xedges = xedges[:-1]
    yedges = yedges[:-1]
    x1_idx = np.argmin(np.abs(xedges - pos[0]))
    y1_idx = np.argmin(np.abs(yedges - pos[1]))
    grid_coord = (int(x1_idx), int(y1_idx))
    return grid_coord


def grid2world(xedges, yedges, pos):
    if (type(pos) == list) or (type(pos) == np.ndarray):
        world_pos = [(float(xedges[x[0]]), float(yedges[x[1]])) for x in pos]
    else:
        world_pos = (float(xedges[pos[0]]), float(yedges[pos[1]]))
    return world_pos

def find_nearest_free(pos, counts):
    # find the nearest free square
    ret_val = pos
    movements = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]
    free_squares = []
    grid_shape = counts.shape

    for k in range(1, 3):
        for movement in movements:    
            new_row = max(min(k*movement[0] + pos[0], grid_shape[0]-1), 0)
            new_col = max(min(k*movement[1] + pos[1], grid_shape[1]-1), 0)
            if (counts[new_row, new_col] == 0):
                free_squares.append((new_row, new_col))

    if len(free_squares) > 0:
        ret_val = free_squares[0]
    
    return ret_val


def bresenham_line(x1, y1, x2, y2):
    """
    Generates a list of (x, y) coordinates representing a line segment
    between (x1, y1) and (x2, y2) using Bresenham's algorithm.
    """
    points = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    x, y = x1, y1

    while True:
        points.append((x, y))
        if x == x2 and y == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return points
