import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from transformers import GPT2Tokenizer
from transformer import Transformer
from safetensors.torch import save_file, load_file
from datasets import load_dataset
import math
import time
import os
import json

class StoryDataset(Dataset):
    def __init__(self, stories, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        for story in stories:
            tokens = tokenizer.encode(story, truncation=True, max_length=max_length)
            if len(tokens) > 1:  # Need at least 2 tokens for input/target pairs
                self.examples.append(torch.tensor(tokens, dtype=torch.long))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        tokens = self.examples[idx]
        # Input: all tokens except last, Target: all tokens except first
        return tokens[:-1], tokens[1:]

def collate_fn(batch, pad_token_id):
    inputs, targets = zip(*batch)
    # Pad sequences to same length in batch using consistent padding token
    inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=pad_token_id)  # type: ignore
    targets = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=pad_token_id)  # type: ignore
    return inputs, targets

def train_epoch(model, dataloader, optimizer, device, tokenizer, scaler):
    model.train()
    total_loss = 0
    num_batches = len(dataloader)

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        # Use mixed precision for forward pass
        with autocast():
            outputs = model(inputs)

            # Reshape for cross entropy: (batch_size * seq_len, vocab_size)
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)

            loss = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)(outputs, targets)

        # Scaled backward pass for mixed precision
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        # Print progress every 64 batches
        if (batch_idx + 1) % 64 == 0:
            print(f"Batch {batch_idx + 1}/{num_batches}, Loss: {loss.item():.4f}")

    return total_loss / num_batches

def main():
    # Device setup - prioritize CUDA, then MPS, then CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA GPU: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Tokenizer setup
    tokenizer = GPT2Tokenizer.from_pretrained('./gpt2-tokenizer-10k')
    # Add a distinct pad token so EOS remains in the loss
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    vocab_size = len(tokenizer.get_vocab())  # Use vocabulary including new pad token
    print(f"Using tokenizer vocab size: {vocab_size} (includes new pad token)")
    print(f"EOS token ID: {tokenizer.eos_token_id}, PAD token ID: {tokenizer.pad_token_id}")

    # Model setup (from TINYSTORIES_RECIPE.md)
    model = Transformer(
        vocab_size=vocab_size,
        d_model=128,
        n_heads=2,
        n_layers=8,
        d_ff=512,
        max_seq_len=512,
        dropout=0.0
    ).to(device)

    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

    # Load TinyStories dataset
    print("Loading TinyStories dataset...")
    dataset = load_dataset("roneneldan/TinyStories", split="train")

    # Take first 64K stories for initial training
    subset_size = 65536
    stories = [story["text"] for story in dataset.select(range(subset_size))] # type: ignore
    print(f"Loaded {len(stories)} stories from TinyStories dataset")

    # Show a sample story
    print(f"\nSample story:")
    print(f"'{stories[0][:200]}...'")
    print(f"Length: {len(stories[0])} characters")

    # Dataset and dataloader
    story_dataset = StoryDataset(stories, tokenizer)
    # Create collate_fn with consistent padding token
    collate_with_padding = lambda batch: collate_fn(batch, pad_token_id=tokenizer.pad_token_id)
    dataloader = DataLoader(story_dataset, batch_size=16, shuffle=True, collate_fn=collate_with_padding)

    # Optimizer and mixed precision scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler()

    # Check for existing checkpoint
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    start_epoch = 0
    best_loss = float('inf')

    # Find latest checkpoint
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.safetensors')]
    if checkpoint_files:
        # Sort by epoch number, not alphabetically
        latest_checkpoint = sorted(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)

        print(f"Found checkpoint: {latest_checkpoint}")

        # Load model state
        model_state = load_file(checkpoint_path)
        model.load_state_dict(model_state)

        # Load training metadata
        metadata_path = checkpoint_path.replace('.safetensors', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                start_epoch = metadata['epoch'] + 1
                best_loss = metadata['best_loss']

        # Load optimizer state (saved separately since safetensors doesn't support it)
        optimizer_path = checkpoint_path.replace('.safetensors', '_optimizer.pt')
        if os.path.exists(optimizer_path):
            optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))

        print(f"Resuming from epoch {start_epoch}, best loss: {best_loss:.4f}")

    # Training loop - run indefinitely until Ctrl-C
    epoch = start_epoch
    try:
        while True:
            print(f"\nEpoch {epoch+1}")
            start_time = time.time()

            avg_loss = train_epoch(model, dataloader, optimizer, device, tokenizer, scaler)

            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s, Average Loss: {avg_loss:.4f}")

            # Save checkpoint after each epoch
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.safetensors')
            save_file(model.state_dict(), checkpoint_path)

            # Save training metadata
            metadata = {
                'epoch': epoch,
                'loss': avg_loss,
                'best_loss': min(best_loss, avg_loss),
                'model_config': {
                    'vocab_size': vocab_size,
                    'd_model': 128,
                    'n_heads': 2,
                    'n_layers': 8,
                    'd_ff': 512,
                    'max_seq_len': 512
                }
            }

            metadata_path = checkpoint_path.replace('.safetensors', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            # Save optimizer state
            optimizer_path = checkpoint_path.replace('.safetensors', '_optimizer.pt')
            torch.save(optimizer.state_dict(), optimizer_path)

            print(f"Checkpoint saved: {checkpoint_path}")

            # Update best loss
            best_loss = min(best_loss, avg_loss)
            epoch += 1

    except KeyboardInterrupt:
        print(f"\nTraining interrupted at epoch {epoch}. Model saved to latest checkpoint.")

    # Save final model
    save_file(model.state_dict(), 'tinystories_model_final.safetensors')
    print("Final model saved in safetensors format!")

if __name__ == "__main__":
    main()