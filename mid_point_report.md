# Introduction  

## Literature Review  

Accurate classification of cancer types based on RNA-sequencing data is a crucial task in computational oncology, with significant implications for diagnosis. Our goal is to use ML to predict the type of cancer based on RNA-sequencing data. This classification is essential in the clinic to support and validate histology and will help guide the treatment of patients.  

Previous studies have laid a strong groundwork for us to expand upon. Jaskowiak et al. compared four clustering methods and twelve distance measures and concluded that **k-medoids** and **hierarchical clustering with average linkage** were superior over complete or single linkage [1]. Moreover, Freyhult et al. pinpointed that **preprocessing steps in using RNA-seq data** can majorly influence performance [2]. Since we are working with high-dimensional data, we aim to test multiple **feature reduction techniques**, similar to those employed by David Kallberg’s group, which analyzed **eleven such techniques** [3].  