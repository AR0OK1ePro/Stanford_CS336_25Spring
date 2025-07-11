import os
from cs336_basics.tokenizer import Tokenizer

def analyze_tokenizer(tokenizer: Tokenizer, top_k: int = 5) -> None:
    """
    Load vocab and merges from files, and print analysis such as:
    - Longest token
    - Top-K longest tokens
    - Vocab and merge sizes
    """
    print(f"📦 Total tokenizer.vocab size: {len(tokenizer.vocab)}")
    print(f"🔗 Total tokenizer.merges: {len(tokenizer.merges)}\n")

    # Find longest tokens
    longest_tokens = sorted(tokenizer.vocab.items(), key=lambda kv: len(kv[1]), reverse=True)[:top_k]

    print(f"🔍 Top {top_k} longest tokens (by byte length):")
    for idx, token in longest_tokens:
        try:
            decoded = token.decode("utf-8")
        except UnicodeDecodeError:
            decoded = "<invalid utf-8>"
        print(f" - ID {idx}: {token} (len={len(token)} bytes) → {decoded}")

def validate_tokenizer(tokenizer: Tokenizer, valid_path: str):
    for i in range(len(tokenizer.merges)):
        assert tokenizer.merges[i][0] + tokenizer.merges[i][1] == tokenizer.vocab[i + 257], f"i = {i}, tokenizer.merges[{i}]: {tokenizer.merges[i]}, tokenizer.vocab: {tokenizer.vocab[i + 257]}"

    with open(valid_path, encoding="utf-8") as f:
        data = f.read()
    assert data == tokenizer.decode(tokenizer.encode(data))

if __name__ == "__main__":
    read_path = os.path.join("tokenizers", "tokenizer_readable-TinyStories_train-v10000.pkl")
    valid_path = os.path.join("data", "TinyStoriesV2-GPT4-valid.txt")
    tokenizer = Tokenizer(vocab={}, merges=[])
    tokenizer.from_file(read_path, special_tokens=["<|endoftext|>"])
        
    analyze_tokenizer(tokenizer, top_k=5)
    validate_tokenizer(tokenizer, valid_path)
