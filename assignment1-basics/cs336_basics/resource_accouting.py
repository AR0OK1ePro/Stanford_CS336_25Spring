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

def AdamW_accouting(batch_size, vocab_size, context_length, num_layers, d_model, num_heads):
    d_ff = 4 * d_model
    # Peak memory:

    # Parameters:
    embedding_params = vocab_size * d_model
    transformer_block_params = d_model * 2 + d_model * d_model * 4 + d_model * d_ff * 3
    final_linear_params = d_model * vocab_size
    norm_params = d_model  # negligible

    param_count = embedding_params + num_layers * transformer_block_params + final_linear_params + norm_params
    parameter_memory_bytes = param_count * 4  # float32
    # Activations:
    activation_embedding = batch_size * context_length * d_model
    activation_norm = activation_embedding
    activation_qkv_proj = 3 * activation_embedding
    activation_qk_mul = num_heads * batch_size * context_length * context_length
    activation_softmax = activation_qk_mul
    activation_weighted_sum_of_v = activation_embedding
    activation_output_proj = activation_embedding
    activation_ffn = 3 * batch_size * context_length * d_model
    activation_transformer_block = activation_norm * 2 + activation_qkv_proj + activation_qk_mul + activation_softmax + activation_weighted_sum_of_v + activation_output_proj + activation_ffn
    activation_logits = batch_size * context_length * vocab_size

    activation_count = activation_embedding + activation_transformer_block * num_layers + activation_norm + activation_logits
    activation_memory_bytes = activation_count * 4 # float32

    # Gradients:
    gradient_memory_bytes = parameter_memory_bytes # float32

    # Optimizer state:
    optimizer_state_memory_bytes = parameter_memory_bytes * 2 # float32
    
    return [parameter_memory_bytes, activation_memory_bytes, gradient_memory_bytes, optimizer_state_memory_bytes]


if __name__ == "__main__":
    # Read arguments from user input
    vocab_size = int(input("Enter vocab size (e.g., 50257): "))
    context_length = int(input("Enter context length (e.g., 1024): "))
    num_layers = int(input("Enter number of layers (e.g., 48): "))
    d_model = int(input("Enter model dimension d_model (e.g., 1600): "))
    num_heads = int(input("Enter number of heads (e.g., 25): "))
    d_ff = int(input("Enter feed-forward dimension d_ff (e.g., 6400): "))
    batch_size = int(input("Enter batch size (e.g., 8): "))

    # FLOPs and parameter stats
    params, mem_bytes, flops, flops_pct = gpt2_custom_param_flops(
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff
    )

    print("=" * 50)
    print("Transformer Model FLOPs and Parameter Stats")
    print("=" * 50)
    print(f"Total parameters: {params:,} (~{params/1e9:.2f}B)")
    print(f"Parameter memory requirement: {mem_bytes/1e9:.2f} GB (float32)")
    print(f"FLOPs per forward pass: {flops/1e12:.2f} T")
    print("\nFLOPs breakdown (%):")
    for k, v in flops_pct.items():
        print(f"{k:>12}: {v:.2f}%")

    # AdamW memory accounting
    memory_parts = AdamW_accouting(
        batch_size=batch_size,
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads
    )
    part_names = ["Parameter", "Activation", "Gradient", "Optimizer state"]
    total_memory = sum(memory_parts)
    print("\n" + "=" * 50)
    print("AdamW Memory Usage Breakdown")
    print("=" * 50)
    for name, mem in zip(part_names, memory_parts):
        print(f"{name:<18}: {mem/1e9:.3f} GB ({mem/total_memory*100:.2f}%)")
    print(f"{'-'*40}")
    print(f"{'Total Memory':<18}: {total_memory/1e9:.3f} GB")

    # Print algebraic expressions for each memory part
    print("\n" + "=" * 50)
    print("Algebraic Expressions for Memory Usage")
    print("=" * 50)
    print("Let:")
    print("  B = batch_size")
    print("  V = vocab_size")
    print("  S = context_length")
    print("  L = num_layers")
    print("  D = d_model")
    print("  H = num_heads")
    print("  d_ff = d_ff (usually 4*D)")
    print("  All memory in bytes (float32: 4 bytes per value)\n")

    print("Parameter memory:")
    print("  (V*D + L*(2*D + 4*D*D + 3*D*d_ff) + D*V + D) * 4")
    print("Activation memory:")
    print("  [B*S*D + L*(2*B*S*D + 3*B*S*D + 2*H*B*S*S + B*S*D + B*S*D + 3*B*S*D) + B*S*D + B*S*V] * 4")
    print("Gradient memory:")
    print("  Same as parameter memory")
    print("Optimizer state memory:")
    print("  Parameter count * 2 * 4")
    print("Total memory:")
    print("  Sum of all above")

    # Print GPT-2-XL memory expression wrt. batch_size
    print("\n" + "=" * 50)
    print("GPT-2-XL shaped model's memory expression that only depends on batch_size")
    print("=" * 50)
    print("Let:")
    print("  B = batch_size")
    print("  V = 50257")
    print("  S = 1024")
    print("  L = 48")
    print("  D = 1600")
    print("  H = 25")
    print("  d_ff = d_ff (usually 4*D)")
    print("  All memory in bytes (float32: 4 bytes per value)\n")
    print("Total memory = 34GB + 13.43GB * batch_size")

    # Print the expression of FLOPs one step of AdamW take:
    print("\n" + "=" * 50)
    print("Print the expression of FLOPs one step of AdamW take:")
    print("=" * 50)
    print("Total FLOPs = 17 * Param_count")