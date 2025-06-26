from collections import Counter
from .pretokenization_example import find_chunk_boundaries
from .train_bpe import (
    init_vocab, pretokenize, split_on_special_tokens, parallel_pretokenize_and_count,
    get_pair_freqs, get_most_frequent_pair, merge_token_freqs
    )

# Test whether vocab is correctly initialized with 256 bytes + special tokens
def test_vocab_init():
    special_tokens = ["<|endoftext|>", "<pad>"]
    vocab = init_vocab(special_tokens)

    assert len(vocab) == 258  # 256 + 2 special tokens
    assert vocab[97] == b'a'
    assert vocab[256] == b"<|endoftext|>"
    assert vocab[257] == b"<pad>"
# Test whether chunk boundaries are found and pretokenization works on chunks
def test_chunk_and_pretokenize():
    input_path = "data/TinyStoriesV2-GPT4-valid.txt"
    special_tokens = ["<|endoftext|>"]

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, 2, b"<|endoftext|>")
        assert len(boundaries) >= 2

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            segments = split_on_special_tokens(chunk, special_tokens)
            assert isinstance(segments, list)
            assert all(isinstance(s, str) for s in segments)

            for segment in segments:
                tokens = list(pretokenize(segment))
                assert all(isinstance(t, bytes) for t in tokens)

# Test whether parallel pretokenization returns a valid token frequency counter
def test_parallel_pretokenize():
    input_path = "data/TinyStoriesV2-GPT4-valid.txt"
    special_tokens = ["<|endoftext|>"]
    token_freqs: Counter[tuple[bytes]] = parallel_pretokenize_and_count(
        input_path=input_path,
        special_tokens=special_tokens,
        num_processes=4
    )
    print(token_freqs.most_common(10))

    assert isinstance(token_freqs, Counter)
    assert len(token_freqs) > 0
    most_common = token_freqs.most_common(5)
    assert all(isinstance(pair, tuple) and isinstance(pair[0], tuple) and isinstance(pair[1], int) for pair in most_common)

def test_merge_step():
    input_path = "data/TinyStoriesV2-GPT4-valid.txt"
    special_tokens = ["<|endoftext|>"]
    
    # 1. 并行预分词统计
    token_freqs: Counter[tuple[bytes]] = parallel_pretokenize_and_count(
        input_path=input_path,
        special_tokens=special_tokens,
        num_processes=2
    )

    # 2. 计算 pair 频率
    pair_freqs = get_pair_freqs(token_freqs)
    assert isinstance(pair_freqs, Counter)
    assert len(pair_freqs) > 0

    print("\nTop 5 pair frequencies:")
    for pair, count in pair_freqs.most_common(5):
        pair_str = (pair[0] + b'' + pair[1]).decode("utf-8", errors="ignore")
        print(f"{pair_str!r}: {count}")

    # 3. 选择最高频 pair
    best_pair = get_most_frequent_pair(pair_freqs)
    assert isinstance(best_pair, tuple)
    assert len(best_pair) == 2
    assert all(isinstance(x, bytes) for x in best_pair)

    print(f"\nBest pair to merge: {(best_pair[0] + best_pair[1]).decode('utf-8', errors='ignore')!r}")

    # 4. 执行一次合并
    new_token_freqs = merge_token_freqs(token_freqs, best_pair)
    assert isinstance(new_token_freqs, Counter)
    assert len(new_token_freqs) > 0

    # 5. 打印 top merge 结果
    print("\nTop 10 merges after first merge step:")
    for token, count in new_token_freqs.most_common(10):
        token_str = b"".join(token).decode("utf-8", errors="ignore")
        print(f"{token}: {token_str!r}: {count}")