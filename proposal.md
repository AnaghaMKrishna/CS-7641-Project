---
layout: default
---

# Project Proposal

## Introduction

### Literature Review
Accurate classification of cancer types based on RNA-sequencing data is a crucial task in computational oncology, with significant implications for diagnosis. Our goal is to use ML to predict the type of cancer based on RNA-sequencing data. This classification is essential in the clinic to support and validate histology and will help guide the treatment for patients. Previous studies have laid a strong groundwork for us to expand upon. Jaskowiak et al. compared 4 clustering methods and 12 distance measures and concluded that k-medoids and hierarchical clustering with average linkage were superior over complete or single linkage[1]. Moreover, Freyhult et al. pinpointed that the preprocessing steps in using RNAseq data can majorly influence the performance[2]. Since we are working with high dimensional data, we aim to test multiple feature reduction techniques like those employed by David Kallberg’s group analyzing 11 such techniques[3].

### Dataset Description
Our dataset contains 6 files containing gene expression profiles of 6 cancer types, present in sample x genes format with 2,952 patient samples and 20,531 genes. The values represent the counts of gene products in the sample.


### Dataset link
[Dataset](http://zenodo.org/records/8192916)[4][5]


## Problem Definition

### Problem
Diagnosing cancer and correct subtype classification from samples collected from patients is crucial in effectively treating cancer. To aid this time-sensitive task, we propose to apply ML techniques for accurately categorizing cancer types by analyzing expression levels of cancer type-specific *signature* genes as a supplementary tool for the clinicians to accelerate diagnosis procedure.


### Motivation
Humans have ~20,000 protein-coding genes. Using manual techniques to identify differentially expressed(DE) genes among conditions is impractical, necessitating the use of sophisticated models to churn this very-high dimensional data. Moreover, DE gene identification is an important step in bioinformatics pipelines for downstream analyses.


## Methods

### Data Preprocessing methods
* **Merging datasets**: Combine datasets based on common features, removing columns with no matching genes.
* **PCA**: Reduces dimensionality in RNA expression data, preventing overfitting and redundancy.
* **Normalization**: Ensures all RNA expression types are on the same scale.
* **Data Imputation**: Fills in missing values to maintain dataset robustness.


### ML Models
* **Supervised**: 
  * **Random Forest**: Excels with high-dimensional data, identifies key genes via Gini and Permutation Importance, and reduces dimensionality by selecting relevant biomarkers, improving model performance.
* **Unsupervised**:
  * **GMM clustering**: Calculates responsibility matrix to give a probability of how much a particular gene expression matches up to all the  various cancer subtypes (which could each be their own gaussian clusters).
  * **Hierarchical clustering**: A good hard clustering technique because cancer subtypes can be categorized both generally and specifically. 
Categorization can be easily visualized using dendrograms, which allow us to see similarities in gene expression at various spectrums.
* **Deep Learning**:
  * **Neural Networks**: Captures non-linear gene relationships, automatically extracts key features, and scales well with architectures like CNNs or autoencoders. Regularization techniques like dropout and batch normalization prevent overfitting and handle noisy data, making them highly effective for cancer classification.


## Results and Discussion

### Quantitative Metrics 
* **Micro F1**
* **One-vs-Rest multiclass ROC**
* **Silhouette Coefficient**
* **Balanced Accuracy**


### Project goals
This project aims to improve the diagnostic accuracy and potentially discover novel patient subclusters within cancer types to develop personalized treatments of cancer patients. Sustainable and ethical considerations include avoiding algorithmic bias, and ensuring model transparency. 

### Expected Results
As we have data from 6 types of cancers, we expect to see 6 distinct clusters and reasonable accuracy, F1(>0.85), AUC/ROC(>0.9) for prediction. We want to create expression signatures by finding the genes that represent a cluster to enable targeted sequencing to only sequence a smaller set of genes for faster and cheaper processing times to classify cancer type in the clinic. 

## Video
{% include youtube.html id="AfDw644U36A" %}

## Gantt Chart

If you are unable to view, [please click here](https://gtvault-my.sharepoint.com/:x:/g/personal/akrishna311_gatech_edu/Eadiz29vLKpCnRs4mFdZ65sBznO7p0sx_qCukL3io7JnZw)

<iframe width="800" height="700" frameborder="2" scrolling="no" src="https://1drv.ms/x/c/7413787cefd7dd8b/IQQHojiVklqHS4MdAunuMfn7AcaTi3X3xg2NidJZ7TvC3l4?em=2&wdAllowInteractivity=False&Item='Sheet1'!B2%3ACZ40&wdInConfigurator=True&wdInConfigurator=True"></iframe>

## Contribution

| Name       | Proposal Contributions|        
|:-------------|:------------------|
| Hina Gaur          | Results and Discussion, References, Video recording and content | 
| Anagha Mohana Krishna  | Problem Definition, Gantt chart, Video slide deck, GitHub pages |
| Mridul Anand           | Introduction, Results and Discussion, References, Video script  | 
| Ani Vedartham | Methods, References, GitHub pages |
| Vedu Arya | Methods, References, GitHub pages  |


## References

1. L. Vidman, D. Källberg, and P. Rydén, “Cluster analysis on high dimensional RNA-seq data with applications to cancer research - An evaluation study,” PLOS ONE, vol. 14, no. 12, p. e0219102, Dec. 2019, doi: https://doi.org/10.1371/journal.pone.0219102.
2. ‌E. Freyhult, M. Landfors, J. Önskog, T. R. Hvidsten, and P. Rydén, “Challenges in microarray class discovery: a comprehensive examination of normalization, gene selection and clustering,” BMC Bioinformatics, vol. 11, no. 1, Oct. 2010, doi: https://doi.org/10.1186/1471-2105-11-503.
3. D. Källberg, L. Vidman, and Patrik Rydén, “Comparison of Methods for Feature Selection in Clustering of High-Dimensional RNA-Sequencing Data to Identify Cancer Subtypes,” Frontiers in Genetics, vol. 12, Feb. 2021, doi: https://doi.org/10.3389/fgene.2021.632620.
4. M. J. Goldman et al., “Visualizing and interpreting cancer genomics data via the Xena platform,” Nature Biotechnology, vol. 38, no. 6, pp. 675–678, May 2020, doi: https://doi.org/10.1038/s41587-020-0546-8.
5. J. N. Weinstein et al., “The Cancer Genome Atlas Pan-Cancer analysis project,” Nature Genetics, vol. 45, no. 10, pp. 1113–1120, Sep. 2013, doi: https://doi.org/10.1038/ng.2764.
6. GeeksforGeeks, “F1 Score in Machine Learning,” GeeksforGeeks, Dec. 27, 2023. https://www.geeksforgeeks.org/f1-score-in-machine-learning/

[Back to home](./)
