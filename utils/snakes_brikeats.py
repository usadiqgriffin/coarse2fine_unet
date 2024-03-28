from functools import partial
from typing import Dict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from scipy.ndimage.interpolation import map_coordinates
from scipy.spatial.distance import cdist

'''
Ref: https://github.com/brikeats/Snakes-in-a-Plane
'''


def computeExternalEnergy(pts: np.ndarray, image: np.ndarray) -> float:
    # external energy (favors low values of distance image)
    pts = pts.transpose()
    dist_vals = map_coordinates(image, pts, order=1)
    external_energy = np.sum(dist_vals)
    
    return external_energy

def computeSpacingEnergy(pts: np.ndarray) -> float:
    # spacing energy (favors equi-distant points)
    prev_pts = np.roll(pts, 1, axis=0)
    prev_pts[0,:] = pts[0,:]
    next_pts = np.roll(pts, -1, axis=0)
    next_pts[-1,:] = pts[-1,:]
    displacements = pts - prev_pts
    displacements = displacements[1:]
    point_distances = np.sqrt(displacements[:,0]**2 + displacements[:,1]**2 + displacements[:,2]**2)
    mean_dist = np.mean(point_distances)
    spacing_energy = np.sum((point_distances - mean_dist)**2)

    return spacing_energy

def computeCurvatureEnergy(pts: np.ndarray) -> float:
    prev_pts = np.roll(pts, 1, axis=0)
    prev_pts[0,:] = pts[0,:]
    next_pts = np.roll(pts, -1, axis=0)
    next_pts[-1,:] = pts[-1,:]
    curvature_1d = prev_pts - 2*pts + next_pts
    curvature = (curvature_1d[:,0]**2 + curvature_1d[:,1]**2 + curvature_1d[:,1]**2)
    curvature = curvature[1:-1]
    curvature_energy = np.sum(curvature)

    return curvature_energy

def computeEndPtsEnergy(pts: np.ndarray, pt0: np.ndarray, ptf: np.ndarray) -> float:
    pt0_energy = np.sqrt(np.sum((pts[0,:] - pt0) ** 2))
    ptf_energy = np.sqrt(np.sum((pts[-1,:] - ptf) ** 2))
    end_pts_energy = pt0_energy + ptf_energy

    return end_pts_energy

def computeTotalEnergy(
    pts_ravel: np.ndarray, 
    image: np.ndarray, 
    pt0: np.ndarray,
    ptf: np.ndarray,
    alpha: float = 1, beta: float = 1, gamma: float = 1, delta: float = 1) -> float:

    pts = np.reshape(pts_ravel, newshape=[-1,3])
    total_energy = 0
    total_energy = alpha * computeExternalEnergy(pts, image)
    total_energy += beta * computeSpacingEnergy(pts)
    total_energy += gamma * computeCurvatureEnergy(pts)
    total_energy += delta * computeEndPtsEnergy(pts, pt0, ptf)

    return total_energy

def plot(image: np.ndarray, points: np.ndarray, projection: str = "coronal", projection_fun=np.sum) -> None:

    if projection=="coronal":
        im = image.copy()
        im = projection_fun(im, 1)
        im = np.squeeze(im)
        plt.imshow(im)
        plt.scatter(points[:,2], points[:,0], c="r", marker="x", s=1)
        plt.show()
    elif projection=="axial":
        pass
    elif projection=="sagittal":
        pass
    else:
        raise ValueError

def fitSnake(points: np.ndarray, image: np.ndarray, pt0: np.ndarray, ptf: np.ndarray, options: Dict) -> np.ndarray:

    assert len(points.shape) == 2
    assert points.shape[1] == 3
    assert len(image.shape) == 3
    assert list(pt0.shape) == [3,]
    assert list(ptf.shape) == [3,]
    assert "alpha" in options
    assert "beta" in options
    assert "gamma" in options
    assert "delta" in options
    assert "method" in options
    assert options["method"] in ["BFGS", "CG", "Powell", "Nelder-Mead"]
    assert "maxiter" in options

    cost_function = partial(
        computeTotalEnergy,
        alpha=options["alpha"], 
        beta=options["beta"], 
        gamma=options["gamma"], 
        delta=options["delta"],
        image=image,
        pt0=pt0,
        ptf=ptf
    )
    options["disp"]=True
    res = optimize.minimize(cost_function, points.ravel(), method=options["method"], options=options)
    points_out = np.reshape(res.x, [-1,3])

    return points_out

def compute_distance_metrics(target_locs_mm: np.ndarray, pred_locs_mm: np.ndarray) -> dict:
    
    D = {}
    distances = cdist(target_locs_mm, pred_locs_mm)
    D["dist_target_to_pred_min_mean_mm"] = distances.min(1).mean()
    D["dist_pred_to_target_min_mean_mm"] = distances.min(0).mean()
    D["dist_target_to_pred_min_max_mm"] = distances.min(1).max()
    D["dist_pred_to_target_min_max_mm"] = distances.min(0).max()
    D["dist_hausdorff_mm"] = max(D["dist_target_to_pred_min_max_mm"], D["dist_pred_to_target_min_max_mm"])
    D["dist_min_mean_mm"] = max(D["dist_target_to_pred_min_mean_mm"], D["dist_pred_to_target_min_mean_mm"])
    
    return D
