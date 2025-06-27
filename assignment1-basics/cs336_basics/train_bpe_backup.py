from collections.abc import Iterator
from multiprocessing import Pool
from collections import Counter, defaultdict
from cs336_basics.pretokenization_example import find_chunk_boundaries
import regex as re

def init_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    """
    Initialize the vocabulary
    """
    vocab: dict[int, bytes] = {}

    # Add 0-255 UTF-8 byte
    for i in range(256):
        vocab[i] = bytes([i])

    # Add special tokens(from index 256)
    for i, token in enumerate(special_tokens):
        vocab[256 + i] = token.encode("utf-8")

    return vocab

def split_on_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    """
    Split text by special tokens and get rid of these special tokens.
    """
    if not special_tokens:
        return [text]
    
    pattern = "|".join(re.escape(token) for token in special_tokens)
    return re.split(pattern, text)

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PRETOKENIZER = re.compile(PAT)

def pretokenize(text: str) -> Iterator[bytes]:
    """
    Use regex of GPT-2 to pretokenize text, return UTF-8 bytes tokens.
    """
    for match in PRETOKENIZER.finditer(text):
        yield match.group(0).encode("utf-8")

def process_chunk(args: tuple[str, list[str]]) -> Counter[tuple[bytes]]:
    """
    pretokenize a single chunk
    """
    chunk, special_tokens = args
    counter = Counter()

    for segment in split_on_special_tokens(chunk, special_tokens):
        for token in pretokenize(segment):
            token_tuple = tuple([bytes([b]) for b in token])
            counter[token_tuple] += 1
    
    return counter

def parallel_pretokenize_and_count(
    input_path: str,
    special_tokens: list[str],
    num_processes: int = 6
) -> Counter[tuple[bytes]]:
    """
    Process chunks in a text file parallelly, pretokenize and compute frequency
    return a global_counter(result after merge)
    """
    split_token_bytes = special_tokens[0].encode("utf-8")
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_processes, split_token_bytes)
        
        # The following is a parallel implementation
        chunk_args = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunk_args.append((chunk, special_tokens))

    # Run pre-tokenization on your chunk and store the counts for each pre-token
    with Pool(processes=num_processes) as pool:
        chunk_counters = pool.map(process_chunk, chunk_args)

    total_counter = Counter()
    for counter in chunk_counters:
        total_counter.update(counter)

    return total_counter

def get_pair_freqs(token_freqs: Counter[tuple[bytes]]) -> Counter[tuple[bytes, bytes]]:
    """
    Compute pair frequencies from token_freqs
    """
    pair_freqs: Counter[tuple[bytes, bytes]] = Counter()
    for token, freq in token_freqs.items():
        #for i in range(len(token) - 1):
            #pair_freqs[(token[i], token[i + 1])] += freq
        for a, b in zip(token, token[1:]):
            pair_freqs[(a, b)] += freq
    
    return pair_freqs

def get_most_frequent_pair(pair_freqs: Counter[tuple[bytes, bytes]]) -> tuple[bytes, bytes]:
    """
    Return the most frequent pair (with lexicographic tie-breaking).
    """
    max_freq = max(pair_freqs.values(), default=0)

    # the biggest frequency pair
    candidates = [pair for pair, freq in pair_freqs.items() if freq == max_freq]

    # return the lexicographically greatest
    return max(candidates)

def build_reverse_index(token_freqs: Counter[tuple[bytes]]) -> dict[tuple[bytes, bytes], set[tuple[bytes]]]:
    """Build reverse index: pair -> set of tokens containing that pair"""
    pair_to_tokens = defaultdict(set)
    
    for token in token_freqs:
        for i in range(len(token) - 1):
            pair = (token[i], token[i+1])
            pair_to_tokens[pair].add(token)
    
    return dict(pair_to_tokens)

def merge_token_freqs_and_update_pair_freqs(
    token_freqs: Counter[tuple[bytes]],
    pair_to_merge: tuple[bytes, bytes],
    pair_freqs: Counter[tuple[bytes, bytes]],
    pair_to_tokens: dict[tuple[bytes, bytes], set[tuple[bytes]]]
) -> tuple[Counter[tuple[bytes]], Counter[tuple[bytes, bytes]], dict[tuple[bytes, bytes], set[tuple[bytes]]]]:
    """
    Merge the most frequent pair
    return a new token_freqs
    """
    a, b = pair_to_merge
    merged = a + b

    token_to_process = pair_to_tokens.get(pair_to_merge, set()).copy()

    for token in token_to_process:
        if token not in token_freqs:
            continue
        
        freq = token_freqs[token]
        new_token: list[bytes] = []

        i = 0
        while i < len(token):
            if i < len(token) - 1 and token[i] == a and token[i+1] == b:
                new_token.append(merged)
                i += 2
            else:
                new_token.append(token[i])
                i += 1
        
        new_token = tuple(new_token)

        token_freqs[token] -= freq
        if token_freqs[token] <= 0:
            del token_freqs[token]
        token_freqs[new_token] += freq

        # Update pair frequencies and reverse index
        for l, r in zip(token, token[1:]):
            pair = (l, r)
            pair_freqs[pair] -= freq
            if pair_freqs[pair] <= 0:
                del pair_freqs[pair]
            if pair in pair_to_tokens:
                if token in pair_to_tokens[pair]:
                    pair_to_tokens[pair].discard(token)
                if not pair_to_tokens[pair]:
                    del pair_to_tokens[pair]
        
        for l, r in zip(new_token, new_token[1:]):
            pair = (l, r)
            pair_freqs[pair] += freq
            if pair not in pair_to_tokens:
                pair_to_tokens[pair] = set()
            pair_to_tokens[pair].add(new_token)

    return token_freqs, pair_freqs, pair_to_tokens



def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train the BPE tokenizer
    """

    # Init vocab and merges
    vocab = init_vocab(special_tokens)
    merges = []

    # Pretokenize input parallelly
    num_processes = 6
    token_freqs = parallel_pretokenize_and_count(input_path, special_tokens, num_processes)

    # Build initial data structures
    pair_freqs = get_pair_freqs(token_freqs)
    pair_to_tokens = build_reverse_index(token_freqs)
     
    print(f"Initial tokens: {len(token_freqs)}, pairs: {len(pair_freqs)}")
    
    # Merge
    while(len(vocab) < vocab_size):
        if not pair_freqs:
            break
        best_pair = get_most_frequent_pair(pair_freqs)
        token_freqs, pair_freqs, pair_to_tokens = merge_token_freqs_and_update_pair_freqs(
            token_freqs, best_pair, pair_freqs, pair_to_tokens)
        new_token = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = new_token
        merges.append(best_pair)

        if len(vocab) % 1000 == 0:
            print(f"Vocab size: {len(vocab)}, Active pairs: {len(pair_freqs)}")

    return vocab, merges