#!/usr/bin/env python3
"""
Generate stories using a trained TinyStories model.

Usage:
    python generate.py <model_path> "<prompt>"

Example:
    python generate.py checkpoints/checkpoint_epoch_2.safetensors "Once upon a time there was a girl who"
"""

import sys
import torch
from transformers import GPT2Tokenizer, pipeline
from transformer import Transformer
from safetensors.torch import load_file
import json
import os

def load_model_and_tokenizer(model_path):
    """Load model and tokenizer from safetensors checkpoint."""

    # Set up tokenizer (same as training)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    # Load model weights
    print(f"Loading model from {model_path}...")
    model_state = load_file(model_path)

    # Try to load model config from metadata
    metadata_path = model_path.replace('.safetensors', '_metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            config = metadata.get('model_config', {})
            print(f"Found model config in metadata: {config}")
    else:
        # Fallback to default config
        config = {
            'vocab_size': 50258,  # Original + pad token
            'd_model': 128,
            'n_heads': 2,
            'n_layers': 8,
            'd_ff': 512,
            'max_seq_len': 512
        }
        print("No metadata found, using default config")

    # Create model with correct config
    model = Transformer(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff'],
        max_seq_len=config['max_seq_len'],
        dropout=0.0
    )

    # Load the trained weights
    model.load_state_dict(model_state)
    model.eval()  # Set to evaluation mode

    # Add config attribute that HuggingFace pipeline expects
    from types import SimpleNamespace
    model.config = SimpleNamespace(
        vocab_size=config['vocab_size'],
        n_positions=config['max_seq_len'],
        n_embd=config['d_model'],
        n_layer=config['n_layers'],
        n_head=config['n_heads'],
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

    print(f"Model loaded: {sum(p.numel() for p in model.parameters())} parameters")

    return model, tokenizer

def generate_story(model, tokenizer, prompt, max_length=200, temperature=0.8):
    """Generate a story completion using raw PyTorch."""

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)

    print(f"Generating story from prompt: '{prompt}'")
    print("=" * 50)

    try:
        # Encode the prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Generate tokens one by one
        generated_ids = input_ids[0].tolist()

        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                input_tensor = torch.tensor([generated_ids[-512:]]).to(device)  # Keep last 512 tokens
                outputs = model(input_tensor)

                # Get logits for the last position
                logits = outputs[0, -1, :] / temperature

                # Sample from the distribution
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()

                # Stop if we hit EOS token
                if next_token == tokenizer.eos_token_id:
                    break

                generated_ids.append(next_token)

        # Decode the full generated sequence
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Print the generated story
        print(generated_text)
        print("=" * 50)
        print(f"Generated {len(generated_text)} characters")

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