from helpers import *

if __name__ == '__main__':
    lipid = load_lipids()
    gene = load_genes()

    section_12_lipids = select_section_lipids(lipid)
    section_12_genes = select_section_genes(gene)
