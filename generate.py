#!/usr/bin/env python3
"""
Generate stories using a trained TinyStories model.

Usage:
    python generate.py <model_path> "<prompt>"

Example:
    python generate.py tinystories_trainer_model "Once upon a time there was a girl who"
"""

import sys
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline
import os
import textwrap

def load_model_and_tokenizer(model_path):
    """Load HuggingFace Transformers model and tokenizer."""

    print(f"Loading model from {model_path}...")

    # Load the model and tokenizer using HF Transformers
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)

    model.eval()  # Set to evaluation mode

    print(f"Model loaded: {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Tokenizer vocab size: {len(tokenizer)}")

    return model, tokenizer

def generate_story(model, tokenizer, prompt, max_length=200, temperature=0.8):
    """Generate a story completion using HuggingFace Transformers."""

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)

    print(f"Generating story from prompt: '{prompt}'")
    print("=" * 50)

    try:
        # Encode the prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Generate using HF Transformers built-in generation
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=2,  # Avoid repetitive text
            )

        # Decode the full generated sequence
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # Format the output with nice word wrapping
        wrapped_text = textwrap.fill(generated_text, width=70, break_long_words=False, break_on_hyphens=False)

        # Print the generated story with nice formatting
        print(wrapped_text)
        print("=" * 70)
        print(f"Generated {len(generated_text)} characters ({len(generated_text.split())} words)")

        return generated_text

    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    if len(sys.argv) < 3:
        print("Usage: python generate.py <model_path> \"<prompt>\"")
        print("Example: python generate.py checkpoints/checkpoint_epoch_2.safetensors \"Once upon a time\"")
        sys.exit(1)

    model_path = sys.argv[1]
    prompt = sys.argv[2]

    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        sys.exit(1)

    try:
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(model_path)

        # Generate story
        generate_story(model, tokenizer, prompt)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()