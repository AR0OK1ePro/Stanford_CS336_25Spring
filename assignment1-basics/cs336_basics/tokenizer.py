import pickle
import regex as re
from collections.abc import Iterable, Iterator

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PRETOKENIZER = re.compile(PAT)

class Tokenizer:
    def __init__(self, 
                vocab: dict[int, bytes],
                merges: list[tuple[bytes, bytes]],
                special_tokens: list[str]=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.pretokenizer_pattern = PRETOKENIZER
        
        # build reverse vocab
        self.reverse_vocab = {value: key for key, value in vocab.items()}

        # build merge priority map for efficient lookup
        self.merge_ranks = {(first, second): i for i, (first, second) in enumerate(merges)}

    def from_file(self, read_path: str, special_tokens=None):
        if special_tokens is None:
            special_tokens = []

        with open(read_path, "rb") as f:
            data = pickle.load(f)
        
        self.vocab = data["vocab"]
        self.merges = data["merges"]
        self.special_tokens = special_tokens
        # build reverse vocab
        self.reverse_vocab = {value: key for key, value in self.vocab.items()}
        # build merge priority map for efficient lookup
        self.merge_ranks = {(first, second): i for i, (first, second) in enumerate(self.merges)}
    
    def _pretokenize(self, text: str): # -> list[str]:
        # Handle special_tokens first
        if self.special_tokens:
            # Create patterns to match special tokens
            # Sort special tokens by length (longest first) to handle overlapping tokens correctly
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            special_pattern = '|'.join(re.escape(token) for token in sorted_special_tokens)

            # Split text on special tokens while keeping the delimiters
            parts = re.split(f'({special_pattern})', text)

            tokens = []
            for part in parts:
                if not part:  # Skip empty strings
                    continue
                elif part in self.special_tokens:
                    # This is a special token, keep it as-is
                    tokens.append(part)
                    # yield part
                else:
                    # This is regular text, apply normal pre-tokenization
                    normal_tokens = self.pretokenizer_pattern.findall(part)
                    tokens.extend(normal_tokens)
                    # yield from self.pretokenizer_pattern.findall(part)
            return tokens
        
        else:
            return re.findall(self.pretokenizer_pattern, text)
            # yield from self.pretokenizer_pattern.findall(text)
        
    def _get_pairs(self, token_bytes: tuple[bytes]) -> set[tuple[bytes, bytes]]:
        pairs = set()
        for i in range(len(token_bytes) - 1):
            pairs.add((token_bytes[i], token_bytes[i+1]))
        return pairs
    
    def _apply_merges(self, token: str) -> list[bytes]:
        token_bytes = [bytes([b]) for b in token.encode("utf-8")]
        while True:
            if len(token_bytes) == 1:
                break
            pairs = self._get_pairs(token_bytes)
            # Find the merge with the lowest merge rank (earlist in training)
            bigram = min(pairs, key=lambda pair: self.merge_ranks.get(pair, float('inf')))
            if bigram not in self.merges:
                break
            # Apply the merge
            first, second = bigram
            new_token_bytes = []
            i = 0
            while i < len(token_bytes):
                try:
                    j = token_bytes.index(first, i)
                    new_token_bytes.extend(token_bytes[i:j])
                    i = j
                except ValueError:
                    new_token_bytes.extend(token_bytes[i:])
                    break
                
                if i < len(token_bytes) - 1 and token_bytes[i + 1] == second:
                    new_token_bytes.append(first + second)
                    i += 2
                else:
                    new_token_bytes.append(token_bytes[i])
                    i += 1
            
            token_bytes = new_token_bytes
        
        return token_bytes

    def _encode_token(self, token: str) -> list[int]:
        # check if it's a special_token:
        if token in self.special_tokens:
            return [self.reverse_vocab[token.encode("utf=8")]]
        
        merged_token_bytes = self._apply_merges(token)
        token_ids = [self.reverse_vocab[token] for token in merged_token_bytes]
        return token_ids
    
    def encode(self, text: str) -> list[int]:
        # start = time.time()
        pre_tokens = self._pretokenize(text)
        # print(f"Pretoken number: {len(pre_tokens)}; Unique pretoken number: {len(set(pre_tokens))}")
        # print(f"Pretokenize time: {time.time() - start}")
        token_ids = []
        token_to_ids = {}
        for token in pre_tokens:
            if token not in token_to_ids:
                token_to_ids[token] = self._encode_token(token)
            token_ids.extend(token_to_ids[token])
        # print(f"token_to_ids len: {len(token_to_ids)}")
        # print(f"Encode time: {time.time() - start}")
        return token_ids

            
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        token_to_ids = {}
        for text in iterable:
            for token in self._pretokenize(text):
                if token not in token_to_ids:
                    token_to_ids[token] = self._encode_token(token)
                token_ids = token_to_ids[token]
                yield from token_ids
    
    def decode(self, ids: list[int]) -> str:
        return b"".join([self.vocab[i] for i in ids]).decode("utf-8", errors="replace")