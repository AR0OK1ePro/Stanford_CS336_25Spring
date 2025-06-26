import cProfile
import cs336_basics.train_bpe
import pstats
import json
import os
from datetime import datetime

def generate_bpe_filenames(model_name: str, vocab_size: int, out_dir: str = "tokenizers") -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    vocab_path = os.path.join(out_dir, f"vocab-{model_name}-v{vocab_size}-{timestamp}.json")
    merges_path = os.path.join(out_dir, f"merges-{model_name}-v{vocab_size}-{timestamp}.txt")
    return vocab_path, merges_path

def save_bpe_model(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    vocab_path: str,
    merges_path: str
) -> None:
    """
    Save vocab to vocab_path (as JSON), and merges to merges_path (as text file).
    """
    # store vocab.json
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(
            {str(idx): token.decode("utf-8", errors="replace") for idx, token in vocab.items()},
            f,
            indent=2,
            ensure_ascii=False
        )

    # store merges.txt（BPE style：1st line: #version，1 line 1 merge）
    with open(merges_path, "w", encoding="utf-8") as f:
        f.write("#version: 0.2\n")
        for a, b in merges:
            f.write(f"{a.decode('utf-8', errors='replace')} {b.decode('utf-8', errors='replace')}\n")

if __name__ == "__main__":

    profiler = cProfile.Profile()
    profiler.enable()

    vocab, merges = cs336_basics.train_bpe.train_bpe(
        input_path="data/owt_train.txt",
        vocab_size=32000,
        special_tokens=["<|endoftext|>"]
    )

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.print_stats(20)

    vocab_path, merges_path = generate_bpe_filenames('owt_train', 32000)

    save_bpe_model(vocab, merges, vocab_path, merges_path)