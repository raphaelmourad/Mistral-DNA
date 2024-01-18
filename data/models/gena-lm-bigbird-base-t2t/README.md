---
tags:
- dna
- human_genome
---

# GENA-LM (gena-lm-bigbird-base-t2t)

GENA-LM is a Family of Open-Source Foundational Models for Long DNA Sequences.

GENA-LM models are transformer masked language models trained on human DNA sequence. 

`gena-lm-bigbird-base-t2t` follows the BigBird architecture and its HuggingFace implementation.

Differences between GENA-LM (`gena-lm-bigbird-base-t2t`) and DNABERT:
- BPE tokenization instead of k-mers;
- input sequence size is about 36000 nucleotides (4096 BPE tokens) compared to 512 nucleotides of DNABERT;
- pre-training on T2T vs. GRCh38.p13 human genome assembly.

Source code and data: https://github.com/AIRI-Institute/GENA_LM

Paper: https://www.biorxiv.org/content/10.1101/2023.06.12.544594v1

## Examples

### Load pre-trained model
```python
from transformers import AutoTokenizer, BigBirdForMaskedLM

tokenizer = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bigbird-base-t2t')
model = BigBirdForMaskedLM.from_pretrained('AIRI-Institute/gena-lm-bigbird-base-t2t')
```


### How to load the model to fine-tune it on classification task
```python
from transformers import AutoTokenizer, BigBirdForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bigbird-base-t2t')
model = BigBirdForSequenceClassification.from_pretrained('AIRI-Institute/gena-lm-bigbird-base-t2t')
```

## Model description
GENA-LM (`gena-lm-bigbird-base-t2t`) model is trained in a masked language model (MLM) fashion, following the methods proposed in the BigBird paper by masking 15% of tokens. Model config for `gena-lm-bigbird-base-t2t` is similar to the `google/bigbird-roberta-base`:

- 4096 Maximum sequence length
- 12 Layers, 12 Attention heads
- 768 Hidden size
- sparse config:
    - block size: 64
    - random blocks: 3
    - global blocks: 2
    - sliding window blocks: 3
- 32k Vocabulary size, tokenizer trained on DNA data.

We pre-trained `gena-lm-bigbird-base-t2t` using the latest T2T human genome assembly (https://www.ncbi.nlm.nih.gov/assembly/GCA_009914755.3/). The data was augmented by sampling mutations from 1000-genome SNPs (gnomAD dataset). Pre-training was performed for 1,070,000 iterations with batch size 256.

## Evaluation
For evaluation results, see our paper: https://www.biorxiv.org/content/10.1101/2023.06.12.544594v1

## Citation
```bibtex
@article{GENA_LM,
	author = {Veniamin Fishman and Yuri Kuratov and Maxim Petrov and Aleksei Shmelev and Denis Shepelin and Nikolay Chekanov and Olga Kardymon and Mikhail Burtsev},
	title = {GENA-LM: A Family of Open-Source Foundational Models for Long DNA Sequences},
	elocation-id = {2023.06.12.544594},
	year = {2023},
	doi = {10.1101/2023.06.12.544594},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2023/06/13/2023.06.12.544594},
	eprint = {https://www.biorxiv.org/content/early/2023/06/13/2023.06.12.544594.full.pdf},
	journal = {bioRxiv}
}
```