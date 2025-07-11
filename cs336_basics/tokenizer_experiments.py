import os
import tracemalloc
from cs336_basics.tokenizer import Tokenizer
import time

def timing(f):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = f(*args, **kwargs)
        end = time.perf_counter()
        print(f"{f.__name__} time: {end - start:.6f}s")
        return result
    return wrapper


def analyze_tokenizer(tokenizer: Tokenizer, top_k: int = 5) -> None:
    """
    Load vocab and merges from files, and print analysis such as:
    - Longest token
    - Top-K longest tokens
    - Vocab and merge sizes
    """
    print("=" * 50)
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
    print("=" * 50)

@timing
def validate_tokenizer(tokenizer: Tokenizer, valid_path: str):
    for i in range(len(tokenizer.merges)):
        assert tokenizer.merges[i][0] + tokenizer.merges[i][1] == tokenizer.vocab[i + 257], f"i = {i}, tokenizer.merges[{i}]: {tokenizer.merges[i]}, tokenizer.vocab: {tokenizer.vocab[i + 257]}"

    with open(valid_path, encoding="utf-8") as f:
        data = f.read()
    assert data == tokenizer.decode(tokenizer.encode(data))

if __name__ == "__main__":
    tinystories_read_path = os.path.join("tokenizers", "tokenizer_readable-TinyStories_train-v10000.pkl")
    tinystories_valid_path = os.path.join("data", "TinyStoriesV2-GPT4-valid.txt")
    tinystories_tokenizer = Tokenizer(vocab={}, merges=[])
    tinystories_tokenizer.from_file(tinystories_read_path, special_tokens=["<|endoftext|>"])

    openwebtext_read_path = os.path.join("tokenizers", "tokenizer_readable-owt_train-v32000.pkl")
    openwebtext_valid_path = os.path.join("data", "owt_valid.txt")
    openwebtext_tokenizer = Tokenizer(vocab={}, merges=[])
    openwebtext_tokenizer.from_file(openwebtext_read_path, special_tokens=["<|endoftext|>"])
        
    # analyze_tokenizer(tinystories_tokenizer, top_k=5)
    validate_tokenizer(tinystories_tokenizer, tinystories_valid_path)

    # analyze_tokenizer(openwebtext_tokenizer, top_k=5)
    validate_tokenizer(openwebtext_tokenizer, openwebtext_valid_path)
