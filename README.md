# Mistral-DNA: Mistra large language model for DNA sequences

# Overview

Here is a repo to pretrain Mistral large language model for DNA sequences. 

# Requirements

If you have an Nvidia GPU, then you must install CUDA and cuDNN libraries. See:  
https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html  
https://developer.nvidia.com/cudnn  
Be aware that you should check the compatibility between your graphic card and the versions of CUDA and cuDNN you want to install. 
This is a bit tricky and time consuming!

To know the version of your NVIDIA driver (if you use an NVIDIA GPU) and the CUDA version, you can type:  
```
nvidia-smi
```
The versions that were used here were : 
- NVIDIA-SMI 535.129.03
- Driver Version: 535.129.03
- CUDA Version: 12.2

The models were developed with python and transformer.  

Before installing python and python packages, you need to install python3 (>=3.10.12) (if you don't have it):  
```
sudo apt update
sudo apt install python3-dev python3-pip python3-venv
```

To install pytorch:  
```
pip install torch==1.13.0
```

Other python packages need to be installed:   
```
pip install transformers==4.37.0.dev0 numpy==1.24.4 pandas==1.4.4 sklearn==0.0 datasets==2.14.4 peft==0.7.2.dev0
```

To generate the data, you need to first install R packages using the following command:
```
if (!require("BiocManager", quietly = TRUE))  
install.packages("BiocManager")  
BiocManager::install("BSgenome.Hsapiens.UCSC.hg38")  
BiocManager::install("GenomicRanges")
BiocManager::install("Biostrings")
```

# Generate the data to pretrain the model

First use the R script "scriptR/script_generate_dna_sequences.R" to generate the DNA sequences which will be used for pretraining the model.

# Pretraining the model

Second, use the jupyter notebook "script_pretrain_mistral-dna.ipynb" to pretrain Mixtral model on DNA sequences. 


# Contact: 
raphael.mourad@univ-tlse3.fr

