import json
import os

def analyze_vocab(vocab_path: str, merges_path: str, top_k: int = 5) -> None:
    """
    Load vocab and merges from files, and print analysis such as:
    - Longest token
    - Top-K longest tokens
    - Vocab and merge sizes
    """
    # Load vocab.json
    with open(vocab_path, "r", encoding="utf-8") as vf:
        vocab_json = json.load(vf)
    vocab = {int(k): v.encode("utf-8", errors="replace") for k, v in vocab_json.items()}

    # Load merges.txt (skip version line)
    with open(merges_path, "r", encoding="utf-8") as mf:
        lines = mf.readlines()
        merges = [
            tuple(line.strip().split(" ", 1))
            for line in lines[1:] if " " in line
        ]

    print(f"📦 Total vocab size: {len(vocab)}")
    print(f"🔗 Total merges: {len(merges)}\n")

    # Find longest tokens
    longest_tokens = sorted(vocab.items(), key=lambda kv: len(kv[1]), reverse=True)[:top_k]

    print(f"🔍 Top {top_k} longest tokens (by byte length):")
    for idx, token in longest_tokens:
        try:
            decoded = token.decode("utf-8")
        except UnicodeDecodeError:
            decoded = "<invalid utf-8>"
        print(f" - ID {idx}: {token} (len={len(token)} bytes) → {decoded}")

if __name__ == "__main__":
    vocab_path = os.path.join("tokenizers", "vocab-owt_valid-v5000-20250626-030104.json")
    merges_path = os.path.join("tokenizers", "merges-owt_valid-v5000-20250626-030104.txt")

    analyze_vocab(
    vocab_path,
    merges_path,
    top_k=5
)