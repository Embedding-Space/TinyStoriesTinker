import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer
from transformer import Transformer
from safetensors.torch import save_file
import math
import time

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

def train_epoch(model, dataloader, optimizer, device, tokenizer):
    model.train()
    total_loss = 0
    num_batches = len(dataloader)

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        # Reshape for cross entropy: (batch_size * seq_len, vocab_size)
        outputs = outputs.view(-1, outputs.size(-1))
        targets = targets.view(-1)

        loss = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)(outputs, targets)  # Ignore padding
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}")

    return total_loss / num_batches

def main():
    # Device setup
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Tokenizer setup
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
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

    # TODO: Load actual TinyStories dataset (for now, dummy data)
    dummy_stories = [
        "Once upon a time, there was a little girl who loved to play.",
        "The cat sat on the mat and looked around the room.",
        "Tommy went to the park with his mom and dad."
    ] * 100  # Repeat to make a small test set

    # Dataset and dataloader
    dataset = StoryDataset(dummy_stories, tokenizer)
    # Create collate_fn with consistent padding token
    collate_with_padding = lambda batch: collate_fn(batch, pad_token_id=tokenizer.pad_token_id)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_with_padding)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        start_time = time.time()

        avg_loss = train_epoch(model, dataloader, optimizer, device, tokenizer)

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s, Average Loss: {avg_loss:.4f}")

    # Save model
    save_file(model.state_dict(), 'tinystories_model.safetensors')
    print("Model saved in safetensors format!")

if __name__ == "__main__":
    main()