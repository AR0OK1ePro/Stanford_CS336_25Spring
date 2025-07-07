def gpt2_custom_param_flops(vocab_size, context_length, num_layers, d_model, num_heads, d_ff):
    """
    Estimate parameter count and FLOPs for a GPT-2-style transformer using user's custom formula.

    Returns:
    - param_count: Total number of parameters
    - memory_bytes: Total memory usage in bytes (assuming float32)
    - flops_total: Total number of FLOPs for one forward pass
    - flops_breakdown_pct: Dictionary with percentage FLOPs per component
    """
    # --- Parameter count ---
    embedding_params = vocab_size * d_model
    transformer_block_params = d_model * 2 + d_model * d_model * 4 + d_model * d_ff * 3
    final_linear_params = d_model * vocab_size
    norm_params = d_model  # negligible

    param_count = embedding_params + num_layers * transformer_block_params + final_linear_params + norm_params
    memory_bytes = param_count * 4  # float32

    # --- FLOPs calculation ---
    qkv_flops = 3 * 2 * context_length * d_model * d_model
    attention_flops = 2 * context_length * context_length * d_model * 2
    output_proj_flops = 2 * context_length * d_model * d_model
    ffn_flops = 3 * 2 * context_length * d_model * d_ff

    # FLOPs per layer
    transformer_block_flops = {
        'qkv': qkv_flops,
        'attention': attention_flops,
        'output_proj': output_proj_flops,
        'ffn': ffn_flops,
    }

    # Total FLOPs (all layers)
    total_transformer_flops = {k: v * num_layers for k, v in transformer_block_flops.items()}
    final_linear_flops = 2 * context_length * d_model * vocab_size

    # Sum up
    flops_total = sum(total_transformer_flops.values()) + final_linear_flops

    # --- FLOPs breakdown statistics ---
    flops_breakdown_pct = {
        k: round(v / flops_total * 100, 2)
        for k, v in total_transformer_flops.items()
    }
    flops_breakdown_pct['final_linear'] = round(final_linear_flops / flops_total * 100, 2)

    return param_count, memory_bytes, flops_total, flops_breakdown_pct

if __name__ == "__main__":
    # Read arguments from user input
    vocab_size = int(input("Enter vocab size (e.g., 50257): "))
    context_length = int(input("Enter context length (e.g., 1024): "))
    num_layers = int(input("Enter number of layers (e.g., 48): "))
    d_model = int(input("Enter model dimension d_model (e.g., 1600): "))
    num_heads = int(input("Enter number of heads (e.g., 25): "))
    d_ff = int(input("Enter feed-forward dimension d_ff (e.g., 6400): "))

    params, mem_bytes, flops, flops_pct = gpt2_custom_param_flops(
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff
    )

    print(f"Total parameters: {params:,} (~{params/1e9:.2f}B)")
    print(f"Memory requirement: {mem_bytes/1e9:.2f} GB (float32)")
    print(f"FLOPs per forward pass: {flops/1e12:.2f} T")
    print("\nFLOPs breakdown (%):")
    for k, v in flops_pct.items():
        print(f"{k:>12}: {v:.2f}%")