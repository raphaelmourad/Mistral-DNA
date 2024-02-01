# Mistral-DNA: Mistral large language model for DNA sequences

# Overview

Here is a repo to pretrain Mistral large language model for DNA sequences. Here the Mixtral model ([Mixtral-8x7B-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)) was modified to significantly reduce the number of parameters mostly by removing layers, such that it could be trained on a GPU such as an RTX3090.

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

The models were developed with python and transformers.  

Before installing python packages, you need to install python3 (>=3.10.12) (if you don't have it):  
```
sudo apt update
sudo apt install python3-dev python3-pip python3-venv
```

Make mistral-dna environment:  
```
conda create -n mitral-dna python=3.8
conda activate mitral-dna
```

To install pytorch:  
```
pip install torch>=1.13.0
```

Other python packages need to be installed:   
```
pip install transformers>=4.37.0.dev0 numpy>=1.24.4 pandas>=1.4.4 sklearn==0.0 datasets>=2.14.4 peft>=0.7.2.dev0
pip install flash-attn==0.2.4
pip install accelerate>=0.21.0
pip install bitsandbytes>=0.37.0
pip install progressbar
pip install tensorboard
pip install torch-xla>=2.0
```

You might need to add this to your .bashrc:
```
export LD_LIBRARY_PATH=$PATH:/your_path/mistral-dna/lib/
```

The pretrained model is available here:
```
https://huggingface.co/RaphaelMourad/Mistral-DNA-v0.1
```

To generate the data, you need to first install R packages using the following command (to type inside R):
```
if (!require("BiocManager", quietly = TRUE))  
install.packages("BiocManager")  
BiocManager::install("BSgenome.Hsapiens.UCSC.hg38")  
BiocManager::install("GenomicRanges")
BiocManager::install("Biostrings")
```

# Generate the data to pretrain the model

If you want to pretrain the model using the whole human genome, first use the R script:
- **scriptR/script_generate_dna_sequences.R** to generate the DNA sequences \

You will obtain the following file (too large to be stored on github):
- **data/genome_sequences/hg38/sequences_hg38_200b.csv.gz** (100% of the human genome)

Alternatively, you can skip this step and use smaller files stored on github:
- **data/genome_sequences/hg38/sequences_hg38_200b_small.csv.gz** (10% of the human genome)
- **data/genome_sequences/hg38/sequences_hg38_200b_verysmall.csv.gz** (1% of the human genome)

# Pretraining the model

Second, in the python folder "scriptPython/", you'll find the jupyter notebook:
- **script_pretrain_mistral-dna.ipynb** to pretrain Mixtral model on DNA sequences. \

Select the data you want to pretrain the model on (full data, small data and very small data).

The script can be ran on [Google Colab](https://colab.research.google.com/drive/1gcw_MYiqwB-pbVYHIx8kevx-ZD7sqMxL#scrollTo=JTYKjBrwRSU6).

# Fine-tuning the model for classification

Third, in the python folder "scriptPython/", you'll find the jupyter notebook:
- **script_finetune.ipynb** to finetune the pretrained Mixtral model on a specific classification task. \

To finetune the model, you must provide a dataset to train the model. 
In the notebook, we used the [GUE dataset](https://drive.google.com/file/d/1GRtbzTe3UXYF1oW27ASNhYX3SZ16D7N2/view). 
The folder GUE should be put in the folder 'data'. 

The script can be ran on [Google Colab](https://colab.research.google.com/drive/19AQsrmiCnEfvgHKz7HQ27-vFsHQogrya).

# Contact: 
raphael.mourad@univ-tlse3.fr

