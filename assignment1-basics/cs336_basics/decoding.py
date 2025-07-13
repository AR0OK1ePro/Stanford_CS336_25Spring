import torch
import os
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer import transformer_lm, softmax
from jaxtyping import Float

def nucleus(p_dist: Float[torch.Tensor, "..."], alpha: float) -> torch.Tensor:
    sorted_values, sorted_indices = torch.sort(p_dist, descending=True)
    accumulated_p = 0
    
    for i in range(len(p_dist)):
        accumulated_p += sorted_values[i]
        if accumulated_p >= alpha:
            break
        
    for j in range(len(p_dist)):
        if j <= i:
            p_dist[sorted_indices[j]] /= accumulated_p
        else:
            p_dist[sorted_indices[j]] = 0
    
    return p_dist


def decoding(tokenizer: Tokenizer, lm: transformer_lm, prompt: str, 
             max_tokens: int=None, t: float=0.5, alpha: float=0.8) -> str:
    assert t >= 0 and t <= 1, "Invalid t, should be 0 <= t <= 1..."
    assert alpha >=0 and alpha <= 1, "Invalid alpha, should be 0 <= alpha <= 1..."
    max_tokens = lm.context_length if max_tokens is None else max_tokens
    assert max_tokens >= 0 and max_tokens <= lm.context_length, "Invalid max_tokens, should be 0 <= max_tokens <= lm.context_length..."

    in_tokens = tokenizer.encode(prompt)
    out_tokens = []
    end_token_idx = tokenizer.reverse_vocab[tokenizer.special_tokens[0].encode("utf-8")]

    while end_token_idx not in out_tokens and len(out_tokens) < max_tokens:
        print(f"Response length: {len(out_tokens)}")
        logits = lm.forward(torch.tensor(in_tokens + out_tokens, device=lm.device))
        p_dist = softmax(logits[-1] / t, -1)
        p_dist = nucleus(p_dist, alpha)
        next_token = int(torch.multinomial(p_dist, num_samples=1))
        out_tokens.append(next_token)
    
    return tokenizer.decode(out_tokens)



if __name__ == "__main__":
    tinystories_read_path = os.path.join("tokenizers", "tokenizer_readable-TinyStories_train-v10000.pkl")
    tinystories_tokenizer = Tokenizer(vocab={}, merges=[])
    tinystories_tokenizer.from_file(tinystories_read_path, special_tokens=["<|endoftext|>"])

    lm = transformer_lm(d_model=512, num_heads=16, d_ff=1344, vocab_size=10000, 
                        num_layers=4, context_length=256, theta=1000, device="mps", dtype=torch.float32)

    lm.load_state_dict(torch.load("checkpoints/TinyStories_LM/lr_1e-1/final_checkpoint.pt", map_location=torch.device('mps'))["model"])
    lm = torch.compile(lm, backend="aot_eager")

    prompt = "Once upon a time"

    response = decoding(tinystories_tokenizer, lm, prompt, 100, 0.3, 0.8)
    print(prompt + response)