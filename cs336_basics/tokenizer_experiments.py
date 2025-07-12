import os
from cs336_basics.tokenizer import Tokenizer
import time
import numpy as np

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

def validate_tokenizer(tokenizer: Tokenizer, valid_path: str):
    for i in range(len(tokenizer.merges)):
        assert tokenizer.merges[i][0] + tokenizer.merges[i][1] == tokenizer.vocab[i + 257], f"i = {i}, tokenizer.merges[{i}]: {tokenizer.merges[i]}, tokenizer.vocab: {tokenizer.vocab[i + 257]}"

    with open(valid_path, encoding="utf-8") as f:
        data = f.read()
    assert data == tokenizer.decode(tokenizer.encode(data))

def experiments(tinystories_tokenizer, openwebtext_tokenizer, tinystories_data_path, openwebtext_data_path):
    ts_samples = []
    with open(tinystories_data_path, encoding="utf-8") as f:
        for i in range(0, 1000000, 100000):
            f.seek(i)
            ts_samples.append(f.read(100000))

    owt_samples = []
    with open(openwebtext_data_path, encoding="utf-8") as f:
        for i in range(0, 1000000, 100000):
            f.seek(i)
            owt_samples.append(f.read(100000))

    ts_ids = []
    owt_ids = []
    for i in range(10):
        ts_ids.append(tinystories_tokenizer.encode(ts_samples[i]))
        owt_ids.append(openwebtext_tokenizer.encode(owt_samples[i]))
    
    print(f"tinystories token length: {[len(ids) for ids in ts_ids]}")
    print(f"openwebtext token length: {[len(ids) for ids in owt_ids]}")

    ts_ids = []
    owt_ids = []
    for i in range(10):
        ts_ids.append(tinystories_tokenizer.encode(owt_samples[i]))
        owt_ids.append(openwebtext_tokenizer.encode(ts_samples[i]))
    
    print(f"tinystories token length: {[len(ids) for ids in ts_ids]}")
    print(f"openwebtext token length: {[len(ids) for ids in owt_ids]}")

def encode_data(tokenizer: Tokenizer, data_path: str, save_path: str):
    token_ids = np.array(tokenizer.encode(open(data_path, encoding="utf-8").read()), dtype=np.uint16)
    token_ids.tofile(save_path)

def inspect_npy_file(filepath, preview_elements=5):
    """
    Inspect a .npy file and print detailed information.

    Parameters:
    - filepath (str): Path to the .npy file.
    - preview_elements (int): Number of elements to preview from the flattened array.
    """
    if not os.path.isfile(filepath):
        print(f"❌ File not found: {filepath}")
        return

    try:
        array = np.memmap(filepath, dtype=np.uint16)
    except Exception as e:
        print(f"❌ Failed to load .npy file: {e}")
        return

    print(f"✅ File loaded: {filepath}")
    print(f"🔹 Type:        {type(array)}")
    print(f"🔹 Dtype:       {array.dtype}")
    print(f"🔹 Shape:       {array.shape}")
    print(f"🔹 Size:        {array.size} elements")
    print(f"🔹 Total Bytes: {array.nbytes:,} bytes (~{array.nbytes / 1024**2:.2f} MB)")

    # If it's a numerical array, print min/max
    if np.issubdtype(array.dtype, np.number):
        print(f"🔹 Value range: [{array.min()}, {array.max()}]")

    # Preview first few elements (flattened)
    preview = array.flatten()[:preview_elements]
    print(f"🔹 Preview [{preview_elements} elements]: {preview}")


if __name__ == "__main__":
    tinystories_read_path = os.path.join("tokenizers", "tokenizer_readable-TinyStories_train-v10000.pkl")
    tinystories_valid_path = os.path.join("data", "TinyStoriesV2-GPT4-valid.txt")
    tinystories_data_path = os.path.join("data", "TinyStoriesV2-GPT4-valid.txt")
    tinystories_save_path = os.path.join("data", "TinyStoriesV2-GPT4-valid.npy")
    tinystories_tokenizer = Tokenizer(vocab={}, merges=[])
    tinystories_tokenizer.from_file(tinystories_read_path, special_tokens=["<|endoftext|>"])

    openwebtext_read_path = os.path.join("tokenizers", "tokenizer_readable-owt_train-v32000.pkl")
    openwebtext_valid_path = os.path.join("data", "owt_valid.txt")
    openwebtext_data_path = os.path.join("data", "owt_valid.txt")
    openwebtext_save_path = os.path.join("data", "owt_valid.npy")
    openwebtext_tokenizer = Tokenizer(vocab={}, merges=[])
    openwebtext_tokenizer.from_file(openwebtext_read_path, special_tokens=["<|endoftext|>"])
        
    # analyze_tokenizer(tinystories_tokenizer, top_k=5)
    # validate_tokenizer(tinystories_tokenizer, tinystories_valid_path)
    encode_data(tinystories_tokenizer, tinystories_data_path, tinystories_save_path)
    '''
    Pretoken number: 539309867; Unique pretoken number: 59934
    Pretokenize time: 174.3832778930664
    token_to_ids len: 59934
    Encode time: 470.70757699012756
    '''


    # analyze_tokenizer(openwebtext_tokenizer, top_k=5)
    # validate_tokenizer(openwebtext_tokenizer, openwebtext_valid_path)
    encode_data(openwebtext_tokenizer, openwebtext_data_path, openwebtext_save_path)
    '''
    Pretoken number: 2474152489; Unique pretoken number: 6601893              │
    Pretokenize time: 587.011049747467                                        │
    token_to_ids len: 6601893                                                 │
    Encode time: 3950.417014837265
    '''

    # experiments(tinystories_tokenizer, openwebtext_tokenizer, tinystories_data_path, openwebtext_data_path)
    # inspect_npy_file(tinystories_save_path)
    # inspect_npy_file(openwebtext_save_path)