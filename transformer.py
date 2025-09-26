from transformers import GPT2Config, GPT2LMHeadModel

def create_tinystories_model(vocab_size):
    """Create a GPT-2 model with TinyStories hyperparameters (matches the actual paper!)"""

    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=512,           # max_seq_len / context length
        n_embd=128,                # d_model / hidden size
        n_layer=8,                 # num_layers
        n_head=2,                  # num_heads
        n_inner=512,               # d_ff / feedforward hidden size
        activation_function="gelu",
        resid_pdrop=0.0,           # residual dropout
        embd_pdrop=0.0,            # embedding dropout
        attn_pdrop=0.0,            # attention dropout
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        use_cache=False,           # Don't cache for training
    )

    model = GPT2LMHeadModel(config)
    return model
