from helpers import *


def points_within_radius(lipids, genes, radius):
    """
    Find points with coordinates within a given radius

    Parameters:
    :param lipids: lipids dataset
    :param genes: genes dataset
    :param radius: radius within which points are considered close

    Returns:
    :return: DataFrame containing points from genes that are within the radius of points in lipids
    """
    result_points = []
    result_indices = []

    for index1, row1 in lipids.iterrows():
        for index2, row2 in genes.iterrows():
            # Calculate distance using the Euclidean distance formula
            distance = np.sqrt((row1['y_cff'] - row2['y_cff'])**2 +
                               (row1['z_cff'] - row2['z_cff'])**2)

            # Check if the distance is within the given radius
            if distance <= radius:
                result_points.append(row2)
                result_indices.append(index1)

    # Create a new DataFrame with points within the radius
    result_df = pd.DataFrame(result_points)

    return result_df
