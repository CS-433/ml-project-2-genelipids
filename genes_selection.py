from typing import Callable

import pandas as pd
import numpy as np
from pandas import DataFrame
from scipy.spatial import cKDTree
from scipy.stats import norm


def aggregate_closest_cells(cells: DataFrame, distances: np.ndarray, closest_indices: np.ndarray, aggregation_function: Callable):
    # Build a dataframe containing the aggregate for
    closest_cells = DataFrame(columns=cells.columns[46:-50])
    for i in range(closest_indices.shape[0]):
        aggregate = aggregation_function(cells.iloc[closest_indices[i], 46:-50], distances[i])
        closest_cells.loc[i] = aggregate

    return closest_cells


def find_k_closest(k: int, cells: DataFrame, lipids: DataFrame, aggregation_function: Callable):
    # Create a KDTree for the cells
    cells_coords = cells[['y_ccf', 'z_ccf']].values
    cells_kdtree = cKDTree(cells_coords)

    # Find the k closest cells for each lipid
    distances, indices = cells_kdtree.query(lipids[['y_ccf', 'z_ccf']].values, k=k)

    return aggregate_closest_cells(cells, distances, indices, aggregation_function)


def find_radius_closest(radius: float, cells: DataFrame, lipids: DataFrame, aggregation_function: Callable):
    # Create a KDTree for the cells
    cells_coords = cells[['y_ccf', 'z_ccf']].values
    cells_kdtree = cKDTree(cells_coords)

    # Find the closest cells in a radius for each lipid
    # Set k to the maximum number of neighbors to avoid excluding any combination
    distances, indices = cells_kdtree.query(lipids[['y_ccf', 'z_ccf']].values, k=cells.shape[0],
                                            distance_upper_bound=radius)

    return aggregate_closest_cells(cells, distances, indices, aggregation_function)


def cell_max(cells: DataFrame, distances: np.ndarray):
    return cells.max()


def cell_average(cells: DataFrame, distances: np.ndarray):
    return cells.mean()


def cell_weighted_average(cells: DataFrame, distances: np.ndarray):
    # Weights based on distance, with a penalty for distances greater than the average closest distance
    weights = norm.pdf(distances, 0, np.std(distances))
    weighted_data = cells * weights[:, np.newaxis]
    return weighted_data.sum(axis=0) / weights.sum() if weights.sum() > 0 else np.zeros(
        cells.shape[1])
