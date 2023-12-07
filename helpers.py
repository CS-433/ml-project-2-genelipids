import pandas as pd
import numpy as np

# Paths
lipid_path = 'data/lba_all_pixels_fully_abamapped11062023.h5'
gene_path = 'data/cell_filtered_w500genes.h5'


def load_lipids(linear_scale=True) -> DataFrame:
    """
    Load lipids dataset

    Parameters:
    :param linear_scale: True for a linear scale, False for an exponential scale

    Returns:
    :return: lipids dataset
    """
    # Loading the dataset
    lipids = pd.read_hdf(lipid_path, key="df")

    # Fill in background pixels
    lipids = lipids.fillna(-9.21)

    if linear_scale:
        #  Exponential (the data we imported were logged, you can play with both scales)
        lipids.iloc[:, 3:205] = np.exp(lipids.iloc[:, 3:205].values)

    return lipids


def load_genes() -> DataFrame:
    """
    Load genes dataset

    Parameters:
    :return: gene dataset
    """
    cells = pd.read_hdf(gene_path)

    return cells



def select_section_lipids(lipids, number=12):
    """
    Select lipids data in the given section of the brain

    Parameters:
    :param lipids: lipids dataset
    :param number: section number

    Returns:
    :return: lipids dataset with only the given section
    """
    section = lipids.loc[lipids['Section'] == number]
    return section


def select_section_genes(cells):
    """
    Select genes data in the given section of the brain

    Parameters:
    :param cells: genes dataset

    Returns:
    :return: genes dataset with only the 12 section
    """
    cells_section_12 = cells.loc[(cells['x_ccf'] > 7.4) & (cells['x_ccf'] < 7.8), :]

    return cells_section_12

