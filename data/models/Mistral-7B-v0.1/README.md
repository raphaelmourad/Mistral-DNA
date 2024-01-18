---
license: apache-2.0
pipeline_tag: text-generation
language:
  - en
tags:
- pretrained
inference:
  parameters:
    temperature: 0.7
---

# Model Card for Mistral-7B-v0.1

The Mistral-7B-v0.1 Large Language Model (LLM) is a pretrained generative text model with 7 billion parameters. 
Mistral-7B-v0.1 outperforms Llama 2 13B on all benchmarks we tested.

For full details of this model please read our [paper](https://arxiv.org/abs/2310.06825) and [release blog post](https://mistral.ai/news/announcing-mistral-7b/).

## Model Architecture

Mistral-7B-v0.1 is a transformer model, with the following architecture choices:
- Grouped-Query Attention
- Sliding-Window Attention
- Byte-fallback BPE tokenizer

## Troubleshooting

- If you see the following error:
```
KeyError: 'mistral'
```
- Or:
```
NotImplementedError: Cannot copy out of meta tensor; no data!
```

Ensure you are utilizing a stable version of Transformers, 4.34.0 or newer.

## Notice

Mistral 7B is a pretrained base model and therefore does not have any moderation mechanisms.

## The Mistral AI Team
 
Albert Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, William El Sayed.