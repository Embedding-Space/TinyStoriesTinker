from datasets import load_dataset
from transformers import (
    GPT2Tokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from transformer import create_tinystories_model


def tokenize_batch(batch, tokenizer):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=512,
        return_attention_mask=True,
    )


def prepare_dataset(tokenizer):
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    # Take first 65536 examples - handle different dataset types
    if hasattr(dataset, 'select'):
        dataset = dataset.select(range(65536))
    else:
        dataset = dataset.take(65536)
    tokenized = dataset.map(
        lambda batch: tokenize_batch(batch, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
    )
    return tokenized.with_format("torch")


def main():
    tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-tokenizer-10k")
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    model = create_tinystories_model(len(tokenizer))
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    dataset = prepare_dataset(tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="checkpoints",
        overwrite_output_dir=False,  # Don't overwrite so we can resume
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        num_train_epochs=30,         # Match our original 30 epochs
        learning_rate=1e-4,
        weight_decay=0.01,
        logging_steps=50,
        save_strategy="epoch",       # Save after each epoch
        save_total_limit=5,          # Keep only last 5 checkpoints
        resume_from_checkpoint=True, # Auto-resume from latest checkpoint
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model("tinystories_trainer_model")
    tokenizer.save_pretrained("tinystories_trainer_model")


if __name__ == "__main__":
    main()
