# ML-HD: Exploring machine learning approaches for phenotype prediction of Huntington’s disease

Age of onset of Huntington's disease is currently being predicted by using the CAG trinucleotide expansion as the main predictor, but it does not explain the entire variability of the phenotype. The present study explores the potential of machine learning algorithms trained with a broader set of genetic data to improve the modeling of this remaining unexplained variance. The data used are single nucleotide polymorphisms genotypes from the Enroll-HD dataset.

This repository contains the code generated throughout the project, explained below following the same `code` directory order, and a toy example (which can be found in `data/features`) to test the ML models scripts (inside `regression_models`). Any script which requires the original raw data, the SNP lookup table or the original list of genes included cannot be tested.

## Source code

### Contents

- [vcf_generation](#vcf_generation)
    - [run_plink.sh](#run_plinksh)
    - [transpose_matrix.cpp](#transpose_matrix)
- [biomart_retrieval](#biomart_retrieval)
    - [biomart_snpid_gene_retrieval_v2.R](#biomart_snpid_gene_retrieval_v2r)
- [subsampling](#subsampling)
    - [random_sample_snps_pick.ipynb](#random_sample_snps_pickipynb)
- [data_preprocessing](#data_preprocessing)
    - [data_normality.ipynb](#data_normalityipynb)
    - [mapping_snps_to_GO_v2.py](#mapping_snps_to_gopy)
    - [alternative_prevalence_filtering_test.ipynb](#alternative_prevalence_filtering_testipynb)
    - [alternative_prevalence_colsuming.py](#alternative_prevalence_colsumingpy)
- [regression_models](#regression_models)
    - [ML_models.ipynb](#ml_modelsipynb)
    - [aoo_scaler.py](#aoo_scalerpy)
    - [model_results.ipynb](#model_resultsipynb)
    - [model_results_plots.ipynb](#model_results_plotsipynb)
    - [/individual_models](#individual_models)
        - [data_loading.py](#data_loadingpy)
        - [evaluating_functions.py](#evaluating_functionspy)

### vcf_generation

#### run_plink.sh

To explore regression approaches to our research objective we need features encoded as a table, where each sample is an observation contained in a row, and each feature is a column. The raw enroll data has samples in columns, and SNPs in rows. The first part of the script subsets the vcf files taking only those rows which represent an SNP contained in the look-up table checking by chromosome and position. This look-up table is generated by [biomart_snpid_gene_retrieval_v2.R](#biomart_snpid_gene_retrieval_v2r).

Then these subsetted rows are translated from the allele encoding of type 0/0 to an integer 0, 1 or 2:

| Old allele encoding | New allele encoding |
|:-------------------:|:-------------------:|
| 0/0 | 0 |
| 0/1 | 1 |
| 1/1 | 2 |

Outside the loop that iterates through chromosomes the script generates intermediate files for each action:
1. Concatenates the subsets of all chromosomes -> data/filtered_enroll/chrompos_snps_model3.txt
2. Transposes the numeric part of the matrix to have samples in columns (by running the build generated by [transpose_matrix.cpp](#transpose_matrix)).
3. Translate the SNPs from chromosome and position to the reference SNP ID.
4. Finally writes the sample names and adds sex and CAG repeat length of each sample as the two first features. These rows are obtained from the metadata enroll file.

Headers in each subsetting and concatenating processes are saved and compared to guarantee we are not mixing up samples and SNPs. A sanity check is also run to ensure 0/0 genotype is the majority.

#### transpose_matrix.cpp

Transposes a numeric txt file.

### biomart_retrieval

#### biomart_snpid_gene_retrieval_v2.R

Retrieves all SNPs in the chromosome coordinate windows corresponding to the genes that we wish to include in the analysis, detailed in a txt file. This list is formed by all the genes in selected GO terms plus a list of HD modifiers described in literature.
The script outputs a table with the chromosome, position, reference SNP id, alleles and gene name. This last file was later modified outside this script to filter out those SNPs which were not single nucleotide variants.

### subsampling

#### random_sample_snps_pick.ipynb

Generates random indices for subsetting the feature matrix into toy examples for easy model training. The shape of the subsets is defined by N (sample size) and j (SNPs to be included in the subset). The generated subsets should always follow the proportion of the original matrix to be a good test set. 

The subset samples and snps indeces and names are saved in separate files in `data/features/subsetting`. The subsetting itself is not done in python.

The saved snps indeces always contain the indices of the HD modifiers genes specified in a text file.

### data_preprocessing

#### data_normality.ipynb

Checks the normality of the outcome variable (age of onset):

- Graphically:
    + Histogram of data behind its Gaussian fit.
    + Q-Q plot comparing with Normal distribution.
- Statistically:
    + Shapiro-Wilk, Kolmogorov-Smirnov, and Scipy's normaltest over the raw y vector.
    + Shapiro-Wilk and Kolmogorov-Smirnov over several transformations of the data (log, reciprocal, exponent, box-cox, Yeo-Johnson, standardization, robust scaler).
    + Dividing the total vector in subsets and checking the percentage of subsets where Shapiro-Wilk test can't prove non-normality.

#### mapping_snps_to_GO_v2.sh

Groups SNPs included in feature matrix (contained in `data/features/subsetting/refIDs_row.txt`) into their corresponding genes using the lookupt table from Biomart, and these genes into their GO terms (broad GO terms which were selected for the core genes set). Saves the results in a table with columns SNP, Gene and GO.


#### alternative_prevalence_filtering_test.ipynb

We want to get how many SNPs have a given minimum alternative variant prevalence. That is, we want to set a minimum percentage of samples that have an alternative allele at a specific SNP. Having an alternative allele, considering how we have build the feature matrix, means that the encoding of the genotype at an SNP is either 1 (heterozygous, one of the two nucleotides is the alternative variant) or 2 (that position in both chromosomes contains the alternative variant), or which is the same, is different from 0.

A set of values of minimum prevalence is defined, and the number and set of SNPs which achieve this minimum are saved. 

The number of SNPs that achieve each minimum is then plotted against the minimum prevalence values range. With such plot we can decide 

#### alternative_prevalence_colsuming.py

Filters the input matrix based on a minimum alternative prevalence threshold. Writes the filtered matrix with the same name as input plus a sufix specifying the threshold. 

Input arguments to the file:

- `matrix`: (str) Input (patient,SNP) matrix path.
- `n_samples`: (int) Number of samples in input matrix.
- `n_snps`: (int) Number of SNPs in input matrix.
- `thres`: (float) Minimum alternative prevalence threshold (as percentage between 0 and 1)


### regression_models

#### ML_models.ipynb

Training and testing of different ML regressions with the toy example. Contains feature scaling, train/test separation, balance study of the input data, and model evaluation functions.

#### aoo_scaler.py

Generates the standard scaler for AOO to be used in the models using all AOO values available, and pickles the scaler object to use it in other scripts.

#### model_results.ipynb

Extracts the SNPs contributing to the models. For the Lasso case, it takes the coefficients different from 0, retrieves the feature's SNP, gene and GO term, and assembles a table with the results.

For the boosting methods, feature importance is sorted by gain (in XGBoost) and Gini index (in Random Forest Regressor), and a similar table to the lasso one is assembled and saved similarly.

#### model_results_plots.ipynb

Graphical representation of the results summarized by [model_results.ipynb](#model_results_plotsipynb), producing Manhattan plots which represent the importance of each SNP, having each SNP represented on its relative position inside the corresponding chromosome.

#### /individual_models

Subdirectory containing individual scripts to run for each model, plus two source scripts containing functions to load data and evaluate models, explained below. Each individual script saves the model evaluations, plus pickles the model for further inspection.

#### data_loading.py

Source script containing functions for data loading used in regression models in this same directory. The functions are:

- `read_sparse_X`: reads the feature matrix by chunks and loads it into executing stage as a csr matrix of float32 values.
- `scale_CAG`: performs a MinMax scaling of the sparse matrix CAG column.
- `interaction_computation`: multiplies all SNP genotypes (0, 1, or 2) by the scaled CAG value of the corresponding subject. The input is sparse (), the function densifies it by chunks (number of rows given by `chunk_size` input parameter) to compute the interaction between SNP and CAG, and makes it sparse again.

#### evaluating_functions.py

Source script containing functions to visualize the training of a model (`model_description`) and how well can it predict the test set (`model_metrics`). 

- `model_description`: returns a plot of the input regressor predictions of the train set over their real outcomes. The percentage of deviance explained (D<sup>2</sup>) by the model is the subtitle of the plot.
- `model_metrics`: Evaluates regressor predicting test samples and returning: R<sup>2</sup>, MSE and MAE; and a plot formed by two subplots of true vs predicted AOO and the residuals vs predicted.
        
Actual and predicted values are plotted transformed to their original range if the transformation scaler is passed as argument in both functions.
    

