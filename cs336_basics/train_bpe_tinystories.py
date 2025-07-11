import cProfile
import cs336_basics.train_bpe
import pstats
import json
import os
import pickle

def generate_bpe_filenames(model_name: str, vocab_size: int, out_dir: str = "tokenizers") -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    vocab_path = os.path.join(out_dir, f"vocab-{model_name}-v{vocab_size}.json")
    merges_path = os.path.join(out_dir, f"merges-{model_name}-v{vocab_size}.txt")
    read_path = os.path.join(out_dir, f"tokenizer_readable-{model_name}-v{vocab_size}.pkl")
    return vocab_path, merges_path, read_path

def save_bpe_model(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    vocab_path: str,
    merges_path: str,
    read_path: str
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

    # store tokenizer_readble.pkl
    with open(read_path, "wb") as f:
        pickle.dump({"vocab": vocab, "merges": merges}, f)

if __name__ == "__main__":

    profiler = cProfile.Profile()
    profiler.enable()

    vocab, merges = cs336_basics.train_bpe.train_bpe(
        input_path="data/TinyStoriesV2-GPT4-train.txt",
        vocab_size=10000,
        special_tokens=["<|endoftext|>"]
    )

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.print_stats(20)

    vocab_path, merges_path, read_path = generate_bpe_filenames('TinyStories_train', 10000)

    save_bpe_model(vocab, merges, vocab_path, merges_path, read_path)