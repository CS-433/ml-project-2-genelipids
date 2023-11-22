# Predicting lipid distribution in a brain section from spatial gene expression

### Library used:
- pandas (version)
- numpy (version)
- matplotlib (version)
- seaborn (version)
- scipy (version)
- pycaret[full]
- tables
- ...

### Dataset composition:
#### Lipids
The lipids dataset is composed by 2 229 568 rows and 208 columns.
Once selected only the Section 12, the remaining rows are 94 747.
Each row represent the dot on the mouse brain Section 12 where the abundance of 202 lipids was measured.
Columns x_ccf, y_ccf and z_ccf represent the spatial coordinates of the dot. Since we work only woth Section 12, the x_ccf coordinate is the same for all measurments and not relevant for this project.
The abundances of lipids are presented in columns 4:205, each lipid having a dedicated column. 
Last three columns represent the aligned representation of each measured dot and won't be relevant for this project.
Lipids mesurment points are uniformely distributed across the brain section.

#### Gene expression 
The gene expression dataset is composed by 3 741 416 rows and 596 columns.
Once selected only the Section 12, the remaining rows are 186 090.
Each row represent the dot on the mouse brain Section 12 where the expression level of 500 lipids was measured.
Columns x_ccf, y_ccf and z_ccf represent the spatial coordinates of the dot and has the same scale as the same coordinates in lipids data.
The expression values for each gene are presented in columns (columns 46:545). All other data presented in Gene expression dataset is not relevant for this project.
Gene expression mesurment points are non-uniformely distributed across the brain section. 

![plot](./Images/1.pdf)

### Main steps of the project:

Exploratory Data Analysis
             |
             |
             |
             v
Elimination of useless measurments ---------> Selection of the strategy to associate 
             |                                gene expression measurment points to 
             |                                lipid abundance points
             |                                       |
             v                                       |
Selection & training 202 models                      |
(1 model per lipid) to predict lipid <---------------|
abundance using gene expression data
             |
             |
             |
             v
Model interpretation & analysis





