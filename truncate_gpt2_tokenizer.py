import json
from pathlib import Path
from transformers import GPT2TokenizerFast

TARGET_MERGES = 9_700
BASE_VOCAB = 256
OUTPUT_DIR = Path("gpt2-tokenizer-10k")

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

tmp_dir = OUTPUT_DIR / "tmp-full"
tokenizer.save_pretrained(tmp_dir)

with open(tmp_dir / "vocab.json") as f:
    full_vocab = json.load(f)

with open(tmp_dir / "merges.txt") as f:
    lines = f.read().splitlines()

header, merge_rules = lines[0], lines[1:]
truncated_merge_rules = merge_rules[:TARGET_MERGES]

inverse_vocab = {idx: token for token, idx in full_vocab.items()}
keep_ids = list(range(BASE_VOCAB + TARGET_MERGES))
kept_tokens = [inverse_vocab[i] for i in keep_ids]

for token in [tokenizer.eos_token, tokenizer.pad_token]:
    if token not in kept_tokens:
        kept_tokens.append(token)

new_vocab = {token: i for i, token in enumerate(kept_tokens)}

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
with open(OUTPUT_DIR / "vocab.json", "w") as f:
    json.dump(new_vocab, f, ensure_ascii=False)

with open(OUTPUT_DIR / "merges.txt", "w") as f:
    f.write(header + "\n")
    for merge in truncated_merge_rules:
        f.write(merge + "\n")

new_tokenizer = GPT2TokenizerFast(
    vocab_file=str(OUTPUT_DIR / "vocab.json"),
    merges_file=str(OUTPUT_DIR / "merges.txt"),
)
new_tokenizer.pad_token = tokenizer.pad_token
new_tokenizer.eos_token = tokenizer.eos_token
new_tokenizer.bos_token = tokenizer.eos_token
new_tokenizer.save_pretrained(OUTPUT_DIR)

print(f"New vocab size: {len(new_tokenizer)}")