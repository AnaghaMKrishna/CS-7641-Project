---
layout: default
description: Project proposal
---

# Project Proposal

## Introduction
Accurate classification of cancer types based on RNA-sequencing data is a crucial task in computational oncology, with significant implications for diagnosis. The goal of this project is to use supervised and unsupervised machine learning methods to predict the type of cancer for patients based on RNA-sequencing data. This sort of classification is essential in the clinic to support and validate histology and will help guide the course of treatment for patients with cancer. Previous studies have laid a strong groundwork for us to expand upon. Jaskowiak et al. compared 4 clustering methods and 12 distance measures and concluded that k-medoids and hierarchical clustering with average linkage were in general superior over hierarchical clustering with complete or single linkage[1]. Moreover, Freyhult et al. pinpointed that the preprocessing steps in using RNAseq data can have a major influence over the performance. Since we are working with high dimensional data, we aim to test multiple feature reduction techniques such as those employed by David Kallberg’s group in an analysis of 11 feature reduction techniques on high dimensional cancer data[2].

#### Literature Review
https://ernest-bonat.medium.com/building-machine-learning-clustering-models-for-gene-expression-rna-seq-data-d0e5af10416d

#### Dataset Description
Our dataset contains gene expression profiles of 6 types of cancers abbreviated as BLCA, CESC, HNSC, KIRC, and LGG. There are 6 files, one for each type of cancer totalling about 1.1 Gb of raw counts data. The data for each file is present in Sample x Genes format with 2,952 data points (patient samples) and 20,531 features (genes) in total. The values represent the counts of each gene in the sample after alignment with a reference transcriptome. 

#### Dataset link
Main dataset - http://zenodo.org/records/8192916
Backup dataset - https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+rna+seq



## Problem Defintion

#### Problem
Use gene count data obtained from RNA seq to classify and predict the type of cancers using the gene expression levels for each sample.
Diagnosing cancer and classifying it into the correct subtype from samples collected from patients is crucial in effectively treating cancer. To aid this time-sensitive task, we propose to apply ML techniques for accurately categorizing cancer types by analyzing expression levels of cancer type-specific “signature” genes as a supplementary tool for the clinicians to speed up diagnosis procedure.


#### Motivation
Given we have sequencing data from different samples without metadata, we can use unsupervised methods to cluster samples into different types of cancers.
Humans have 20,000 protein-coding genes. Using manual techniques to identify genes that are differentially expressed(DE) between different conditions is impractical, necessitating the use of sophisticated models to churn this very-high dimensional data. Moreover, identification of DE genes is an important step in bioinformatics pipelines for further downstream analyses.




## Methods

#### Data Preprocessing methods
* Merging datasets: Combine datasets based on common features, removing columns with no matching genes.
* PCA: Reduces dimensionality in RNA expression data, preventing overfitting and redundancy.
* Normalization: Ensures all RNA expression types are on the same scale.
* Data Imputation: Fills in missing values to maintain dataset robustness.


#### ML Models
* Supervised: 
  * Random Forest:  Excels with high-dimensional data, identifies key genes via Gini and Permutation Importance, and reduces dimensionality by     selecting relevant biomarkers, improving model performance.
* Unsupervised:
  * GMM clustering: Can calculate responsibility matrix to give a probability of how much a particular gene expression matches up to all the  various cancer subtypes (which could each be their own gaussian clusters).
  * Hierarchical clustering: A good hard clustering technique because cancer subtypes can be categorized both generally and specifically. 
Categorization can be easily visualized using dendrograms, which will allow us to see similarities in gene expression at various spectrums.
* Deep Learning:
  * Neural Networks: Captures non-linear gene relationships, automatically extracts key features, and scales well with architectures like CNNs or autoencoders. Regularization techniques like dropout and batch normalization prevent overfitting and handle noisy data, making them highly effective for cancer classification.




## Results and Discussion

#### Quantitative Metrics 
* Micro F1 - It will provide a balanced measure of precision and recall as both false positives and false negatives can have serious consequences in cancer diagnosis.
* One-vs-Rest multiclass ROC - plots the True Positive Rate  against the False Positive Rate at different thresholds and will help us understand which cancer types are most easily discriminated against.
* Silhouette Coefficient - It combines information about cohesion and separation and will help determine if the clusters discovered actually correspond to different cancer types.
* Balanced Accuracy - It is an arithmetic mean of sensitivity and specificity and will give equal weight to each cancer type.


#### Project goals
This project aims to improve the diagnostic accuracy and potentially discover novel patient subclusters within cancer types to develop  personalized treatments of cancer patients. Sustainable and ethical considerations include avoiding algorithmic bias, and ensuring model transparency. 

#### Expected Results
As we have data from 6 types of cancers, we expect to see 6 distinct clusters and reasonable accuracy, F1, AUC/ROC for prediction. We expect to have a F1 greater than 0.85 and AUROC greater than 0.9 that match previous attempts to cluster similar datasets. Moreover, we want to create expression signatures by finding the genes that represent a cluster to enable targeted sequencing to only sequence a smaller set of genes. This would allow for faster and cheaper processing times to classify a patient’s cancer type in the clinic. 

## Video

## Gantt Chart
<img width="541" alt="image" src="https://github.gatech.edu/ML7641-Group55/CS-7641-Project/assets/73053/bcd474e4-412f-4e1c-84e4-9313042fbe5c">


## Contribution

| Name       | Proposal Contributions|        
|:-------------|:------------------|
| Hina Gaur          | Results and Discussion, References, Video | 
| Anagha Mohana Krishna  | Problem Definition, References, Video  |
| Mridul Anand           | Introduction, Results and Discussion, References, Video   | 
| Ani Vedartham | Methods, References  |
| Vedu Arya | Methods, References  |


## References

1. L. Vidman, D. Källberg, and P. Rydén, “Cluster analysis on high dimensional RNA-seq data with applications to cancer research - An evaluation study,” PLOS ONE, vol. 14, no. 12, p. e0219102, Dec. 2019, doi: https://doi.org/10.1371/journal.pone.0219102.
2. ‌D. Källberg, L. Vidman, and Patrik Rydén, “Comparison of Methods for Feature Selection in Clustering of High-Dimensional RNA-Sequencing Data to Identify Cancer Subtypes,” Frontiers in Genetics, vol. 12, Feb. 2021, doi: https://doi.org/10.3389/fgene.2021.632620.
3. M. J. Goldman et al., “Visualizing and interpreting cancer genomics data via the Xena platform,” Nature Biotechnology, vol. 38, no. 6, pp. 675–678, May 2020, doi: https://doi.org/10.1038/s41587-020-0546-8.
4. J. N. Weinstein et al., “The Cancer Genome Atlas Pan-Cancer analysis project,” Nature Genetics, vol. 45, no. 10, pp. 1113–1120, Sep. 2013, doi: https://doi.org/10.1038/ng.2764.
5. GeeksforGeeks, “F1 Score in Machine Learning,” GeeksforGeeks, Dec. 27, 2023. https://www.geeksforgeeks.org/f1-score-in-machine-learning/


[Back to home](./)
