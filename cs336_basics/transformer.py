import torch
import math
from jaxtyping import Float, Bool
from einops import rearrange, reduce, einsum

class Linear(torch.nn.Module):
    """
    A custom implementation of a linear (fully connected) layer.
    Applies a linear transformation to the incoming data: y = xA^T.
    """
    def __init__(self, in_features, out_features, device=None, dtype=None):
        """
        Initializes the Linear layer.

        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            device (torch.device, optional): The device to store the parameters on.
            dtype (torch.dtype, optional): The desired data type of the parameters.
        """
        super().__init__()
        self.sigma = torch.tensor(2 / (in_features + out_features), dtype=dtype, device=device).sqrt_()
        # Weight matrix of shape (out_features, in_features)
        self.weight: Float[torch.Tensor, "out_features in_features"] = torch.empty(
            (out_features, in_features), dtype=dtype, device=device
        )
        # Initialize weights with truncated normal distribution
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=self.sigma, a=-3*self.sigma, b=3*self.sigma)
        self.weight = torch.nn.Parameter(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the linear transformation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (..., in_features).

        Returns:
            torch.Tensor: Output tensor of shape (..., out_features).
        """
        return einsum(x, self.weight, 
                      "... in_features, out_features in_features -> ... out_features")

class Embedding(torch.nn.Module):
    """
    A custom implementation of an embedding layer.
    Maps token indices to embedding vectors.
    """
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """
        Initializes the Embedding layer.

        Args:
            num_embeddings (int): Number of embeddings (vocabulary size).
            embedding_dim (int): Dimension of each embedding vector.
            device (torch.device, optional): The device to store the parameters on.
            dtype (torch.dtype, optional): The desired data type of the parameters.
        """
        super().__init__()
        # Embedding matrix of shape (num_embeddings, embedding_dim)
        self.embedding: Float[torch.Tensor, "num_embedding embedding_dim"] = torch.empty(
            (num_embeddings, embedding_dim), dtype=dtype, device=device
        )
        # Initialize embeddings with truncated normal distribution
        torch.nn.init.trunc_normal_(self.embedding, mean=0, std=1, a=-3, b=3)
        self.embedding = torch.nn.Parameter(self.embedding)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Retrieves the embedding vectors for the given token IDs.

        Args:
            token_ids (torch.Tensor): Tensor of token indices, shape (...).

        Returns:
            torch.Tensor: Tensor of embeddings, shape (..., embedding_dim).
        """
        return self.embedding[token_ids]

class RMSNorm(torch.nn.Module):
    """
    Root Mean Square Layer Normalization.
    Normalizes the input over the last dimension and applies a trainable gain.
    """
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        Initializes the RMSNorm layer.

        Args:
            d_model (int): The size of the last dimension to normalize.
            eps (float, optional): Small value to avoid division by zero.
            device (torch.device, optional): The device to store the parameters on.
            dtype (torch.dtype, optional): The desired data type of the parameters.
        """
        super().__init__()
        self.gain: Float[torch.Tensor, "d_model"] = torch.nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies RMS normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (..., d_model).

        Returns:
            torch.Tensor: Normalized tensor of shape (..., d_model).
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # Compute the inverse root mean square for normalization
        RMS_reverse = 1 / (reduce(x**2, "... d_model -> ...", "mean") + self.eps).sqrt()
        # Apply normalization and gain using einsum for correct broadcasting
        result = einsum(x * self.gain, RMS_reverse, "... d_model, ... -> ... d_model")
        return result.to(in_dtype)

class SwiGLU(torch.nn.Module):
    """
    SwiGLU feed-forward block as used in modern transformer architectures.
    Applies two linear projections and a gating mechanism with the SiLU activation.
    """
    def __init__(self, d_model: int, d_ff: int, dtype=None, device=None):
        """
        Initializes the SwiGLU layer.

        Args:
            d_model (int): Input and output feature dimension.
            d_ff (int): Inner feed-forward dimension (should be ~8/3*d_model, rounded to multiple of 64).
            dtype (torch.dtype, optional): The desired data type of the parameters.
            device (torch.device, optional): The device to store the parameters on.
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        # Standard deviation for weight initialization
        self.sigma = torch.tensor(2 / (self.d_model + self.d_ff), dtype=dtype, device=device).sqrt_()
        # First linear projection
        self.weight_1: Float[torch.Tensor, "d_ff d_model"] = torch.nn.Parameter(
            torch.empty((self.d_ff, self.d_model), dtype=dtype, device=device)
        )
        torch.nn.init.trunc_normal_(self.weight_1, 0, self.sigma, a=-3*self.sigma, b=3*self.sigma)
        # Second linear projection (output)
        self.weight_2: Float[torch.Tensor, "d_model d_ff"] = torch.nn.Parameter(
            torch.empty((self.d_model, self.d_ff), dtype=dtype, device=device)
        )
        torch.nn.init.trunc_normal_(self.weight_2, 0, self.sigma, a=-3*self.sigma, b=3*self.sigma)
        # Gating linear projection
        self.weight_3: Float[torch.Tensor, "d_ff d_model"] = torch.nn.Parameter(
            torch.empty((self.d_ff, self.d_model), dtype=dtype, device=device)
        )
        torch.nn.init.trunc_normal_(self.weight_3, 0, self.sigma, a=-3*self.sigma, b=3*self.sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the SwiGLU transformation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (..., d_model).

        Returns:
            torch.Tensor: Output tensor of shape (..., d_model).
        """
        # First linear projection
        hidden = einsum(self.weight_1, x, "d_ff d_model, ... d_model -> ... d_ff")
        # SiLU activation (Swish)
        SiLU = hidden * torch.sigmoid(hidden)
        # Gating projection
        Gate = einsum(self.weight_3, x, "d_ff d_model, ... d_model -> ... d_ff")
        # Output projection
        return einsum(self.weight_2, SiLU * Gate, "d_model d_ff, ... d_ff -> ... d_model")

class RotaryPositionEmbedding(torch.nn.Module):
    """
    Rotary Position Embedding (RoPE) layer.
    Applies rotary positional encoding to input tensors for attention mechanisms.
    """
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        Initializes the RotaryPositionEmbedding layer.

        Args:
            theta (float): Base for computing inverse frequencies.
            d_k (int): Dimension of the key/query vectors (must be even).
            max_seq_len (int): Maximum sequence length for which to cache embeddings.
            device (torch.device, optional): The device to store the buffers on.
        """
        super().__init__()
        assert d_k % 2 == 0
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.theta = theta
        # Compute inverse frequencies for rotary embedding
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device) / d_k))
        pos = torch.arange(max_seq_len, device=device).float().unsqueeze(1)
        angles = pos * inv_freq.unsqueeze(0)
        # Cache cos and sin values for all positions
        self.register_buffer("cos_cached", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cached", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        """
        Applies rotary position embedding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (..., seq_len, d_k).
            token_positions (torch.Tensor): Tensor of token positions, shape (..., seq_len).

        Returns:
            torch.Tensor: Tensor with rotary position embedding applied, shape (..., seq_len, d_k).
        """
        assert x.shape[-1] == self.d_k
        seq_len = x.shape[-2]
        # Reshape last dimension into pairs for rotation
        x = rearrange(x, "... seq_len (d_k1 d_k2) -> ... seq_len d_k1 d_k2", d_k2=2)
        # Select cached cos and sin values for the given positions
        if token_positions is None:
            token_positions = torch.arange(0, seq_len)
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]
        # Apply rotary transformation
        x_1 = x[..., 0]
        x_2 = x[..., 1]
        out = torch.stack([
            x_1 * cos - x_2 * sin,
            x_1 * sin + x_2 * cos
        ], dim=-1)
        # Restore original shape
        return rearrange(out, "... seq_len d_k1 d_k2 -> ... seq_len (d_k1 d_k2)", d_k2=2)

def softmax(x: torch.Tensor, i: int):
    x_ith_max, _ = torch.max(x, dim=i, keepdim=True)
    exp_x = torch.exp(x - x_ith_max)
    exp_x_sum = torch.sum(exp_x, dim=i, keepdim=True)
    return exp_x / exp_x_sum
    
def scaled_dot_product_attention(
        query: Float[torch.Tensor, "batch_size ... seq_len d_k"],
        key: Float[torch.Tensor, "batch_size ... seq_len d_k"],
        value: Float[torch.Tensor, "batch_size ... seq_len d_v"],
        mask: Bool[torch.Tensor, "seq_len seq_len"] = None 
):
    attention = einsum(query, key, "... q d_k, ... k d_k -> ... q k")
    attention /= torch.sqrt(torch.tensor(key.shape[-1], device=key.device, dtype=key.dtype))
    if mask is not None:
        attention = torch.where(mask.bool(), attention, -float('inf'))
    
    attention = softmax(attention, -1)
    return einsum(attention, value, "... q k, ... k d_v -> ... q d_v")

class multihead_self_attention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = None, theta: float = None, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        self.w_q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.w_k = Linear(d_model, d_model, device=device, dtype=dtype)
        self.w_v = Linear(d_model, d_model, device=device, dtype=dtype)
        self.w_o = Linear(d_model, d_model, device=device, dtype=dtype)
        
        if max_seq_len is not None and theta is not None:
            d_k = self.d_model // self.num_heads
            self.rope = RotaryPositionEmbedding(theta, d_k, max_seq_len, device=device)
        else:
            self.rope = None
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        seq_len = x.shape[-2] 

        query = self.w_q.forward(x)
        key = self.w_k.forward(x)
        value = self.w_v.forward(x)


        query = rearrange(query, "batch_size ... seq_len (h d_k)  -> batch_size ... h seq_len d_k", 
                            h = self.num_heads)
        key = rearrange(key, "batch_size ... seq_len (h d_k) -> batch_size ... h seq_len d_k", 
                            h = self.num_heads)
        value = rearrange(value, "batch_size ... seq_len (h d_v) -> batch_size ... h seq_len d_v", 
                            h = self.num_heads)
                            
        if self.rope is not None:
            query = self.rope.forward(query, token_positions)
            key = self.rope.forward(key, token_positions)

        mask = torch.tril(torch.ones(seq_len, seq_len), diagonal=0).bool()

        attn_out = scaled_dot_product_attention(query, key, value, mask)

        attn_out = rearrange(attn_out, "batch_size ... h seq_len d_v -> batch_size ... seq_len (h d_v)")

        return self.w_o.forward(attn_out)

class transformer_block(torch.nn.Module):
    """
    A single transformer block consisting of:
    - Pre-normed multi-head self-attention with residual connection
    - Pre-normed SwiGLU feed-forward with residual connection
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int = None, theta: float = None, device=None, dtype=None):
        """
        Initialize the transformer block.

        Args:
            d_model (int): Model dimension.
            num_heads (int): Number of attention heads.
            d_ff (int): Feed-forward hidden dimension.
            max_seq_len (int, optional): Maximum sequence length for RoPE.
            theta (float, optional): RoPE theta parameter.
            device (torch.device, optional): Device for parameters.
            dtype (torch.dtype, optional): Data type for parameters.
        """
        super().__init__()
        self.rmsnorm_1 = RMSNorm(d_model, device=device)
        self.mha = multihead_self_attention(d_model, num_heads, max_seq_len, theta, device=device, dtype=dtype)
        self.rmsnorm_2 = RMSNorm(d_model, device=device)
        self.ff = SwiGLU(d_model, d_ff, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for the transformer block.

        Args:
            x (torch.Tensor): Input tensor (..., seq_len, d_model).
            token_positions (torch.Tensor, optional): Token positions for RoPE.

        Returns:
            torch.Tensor: Output tensor (..., seq_len, d_model).
        """
        # Apply first RMSNorm, then MHA with residual connection
        activations = self.mha.forward(
            self.rmsnorm_1.forward(x), token_positions) + x
        # Apply second RMSNorm, then FFN with residual connection
        results = self.ff.forward(
            self.rmsnorm_2.forward(activations)) + activations
        return results

class transformer_lm(torch.nn.Module):
    """
    Transformer language model composed of:
    - Token embedding
    - Stack of transformer blocks
    - Final RMSNorm
    - Output linear projection to vocabulary size
    """
    def __init__(self, d_model: int, 
                 num_heads: int, d_ff: int,
                 vocab_size: int,
                 num_layers: int,
                 context_length: int = None,
                 theta: float = None,
                 device=None, dtype=None
                ):
        """
        Initialize the transformer language model.

        Args:
            d_model (int): Model dimension.
            num_heads (int): Number of attention heads.
            d_ff (int): Feed-forward hidden dimension.
            vocab_size (int): Vocabulary size.
            num_layers (int): Number of transformer blocks.
            context_length (int, optional): Maximum sequence length.
            theta (float, optional): RoPE theta parameter.
            device (torch.device, optional): Device for parameters.
            dtype (torch.dtype, optional): Data type for parameters.
        """
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        # Stack of transformer blocks
        self.blocks = torch.nn.ModuleList([
            transformer_block(
                d_model, num_heads, d_ff, context_length, theta, device=device, dtype=dtype 
            ) for _ in range(num_layers)
        ])
        self.rmsnorm = RMSNorm(d_model, device=device, dtype=dtype)
        self.linear = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for the transformer language model.

        Args:
            token_ids (torch.Tensor): Input token indices (batch_size, seq_len).
            token_positions (torch.Tensor, optional): Token positions for RoPE.

        Returns:
            torch.Tensor: Output logits (batch_size, seq_len, vocab_size).
        """
        # Embed input tokens
        x = self.embedding.forward(token_ids)
        # Pass through each transformer block
        for block in self.blocks:
            x = block.forward(x, token_positions)
        # Final RMSNorm and linear projection to vocab size
        logits = self.linear.forward(self.rmsnorm.forward(x))
        # Softmax is applied outside if needed
        result = softmax(logits, -1)

        return logits

