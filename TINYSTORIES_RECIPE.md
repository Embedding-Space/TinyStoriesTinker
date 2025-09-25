# TinyStories Training Recipe

*Based on "TinyStories: How Small Can Language Models Be and Still Speak Coherent English?" (Eldan & Li, 2023)*

## Core Insight
Small models (1-33M params) can generate coherent text when trained on **simplified, constrained data** rather than massive diverse corpora. The key is vocabulary limitation (3-4 year old level) not model scale.

## Architecture
- **Base**: GPT-Neo decoder-only transformer
- **Context length**: 512 tokens
- **Vocabulary**: GPT-2 tokenizer, truncated to top 10K tokens
- **Embedding dimension**: 64-1024 (128-256 sweet spot)
- **Layers**: 1-12 (even 1 layer works!)
- **Heads**: 1-16 (proportional to dimensions)

## Proven Model Sizes
- **1M params**: 1 layer, 64 dim - basic coherence
- **2.5M params**: 8 layers, 128 dim - good stories
- **28M params**: 8 layers, 768 dim - high quality
- **33M params**: 4 layers, 1024 dim - best performance

## Dataset
- **Source**: GPT-3.5/GPT-4 generated stories
- **Constraint**: Simple vocabulary (3-4 year old level)
- **Length**: 2-3 paragraphs per story
- **Diversity**: Random word prompts + story features (dialogue, twist, etc.)
- **Size**: ~2M stories for full dataset

## Training
- **Framework**: Raw PyTorch (no Trainer complexity)
- **Time**: <24 hours on single V100 GPU
- **Loss**: Standard next-token prediction
- **Optimizer**: AdamW (paper doesn't specify hyperparams)

## Key Success Factors
1. **Constrained vocabulary** > model size
2. **Consistent story structure** (beginning/middle/end)
3. **Simple but complete grammar**
4. **Diverse scenarios** within vocabulary constraints

## Our Target Architecture (~2.5M params)
```
d_model: 128
n_heads: 2
n_layers: 8
d_ff: 512
vocab_size: 10000
max_seq_len: 512
```

## Evaluation
Stories should be:
- Grammatically correct
- Contextually consistent
- Creative within constraints
- Multi-paragraph coherent

*"The journey matters more than the destination"*