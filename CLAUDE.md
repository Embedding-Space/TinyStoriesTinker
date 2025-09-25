# TinyStoriesTinker

## Mission
Learn how transformers work by building the simplest working story generator we can. This is educational tinkering, not production ML - we want to understand every component from embeddings to attention heads to autoregressive generation.

## Inspiration
The **TinyStories** paper (Eldan & Li, 2023) showed that tiny transformer models (1-33M parameters) can generate coherent multi-paragraph stories when trained on a carefully crafted dataset of simple vocabulary stories. They proved that "emergence" of language capabilities doesn't require massive scale - just good data and architecture.

## Our Plan
1. **Import tokenizer**: Use GPT-2's tokenizer (truncated to 10K vocab) - this is boring infrastructure
2. **Implement transformer architecture**: Build GPT-Neo style decoder blocks from scratch in PyTorch - this is the educational core
3. **Target architecture**: ~2.5M parameters (128 hidden dim, 8 layers, 2 attention heads)
4. **Write training loop**: Raw PyTorch, no `Trainer` complexity - we want to see every step
5. **Train on TinyStories dataset**: Available on Hugging Face
6. **Test story generation**: Watch our tiny model write coherent narratives

## Learning Philosophy
**Go slow and teach.** This is pedagogical exploration, not problem-solving. The human wants to understand transformer mechanics deeply - attention head specialization, embedding spaces, autoregressive generation, etc. 

**Don't rush ahead** with "efficient" solutions. **Don't treat this as a production ML problem.** Break down concepts, explain the math, show how each piece connects. The goal is understanding, not just working code.

The journey matters more than the destination.

## Technical Details
- Use `uv` exclusively to manage Python packages and to run scripts
- Don't add any code comments unless specifically asked. Part of the exercise will be for Jeffery to go through the code and understand it by adding his own comments.
