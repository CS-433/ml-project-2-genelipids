import pandas as pd
import numpy as np
from run import *


def loading_lipids(linear_scale=True):
    # Loading the dataset
    lipids = pd.read_hdf(lipid_path)

    # Fill in background pixels
    lipids = lipids.fillna(-9.21)

    if linear_scale:
        #  Exponential (the data we imported were logged, you can play with both scales)
        lipids.iloc[:, 3:205] = np.exp(lipids.iloc[:, 3:205].values)

    return lipids


def select_section(lipids, number=12):
    section = lipids.loc[lipids['Section'] == number]
    return section

def compute_euclidian_distance(x_lipid, y_lipid, x_gene, y_gene):
    pass
