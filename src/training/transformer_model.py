import tensorflow as tf
from tensorflow.keras import layers

"""
Indian Sign Language (ISL) Landmark Transformer Model
Matches the 24-layer weight architecture used for the 94% accuracy model.
"""

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    """
    Standard Transformer Encoder block.
    Each block contains:
    - 2 Layer Normalization layers
    - 1 Multi-Head Attention layer
    - 2 Conv1D layers (Feed-Forward network)
    """
    # 1. Normalization and Multi-Head Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # 2. Feed-Forward Network
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    # ff_dim is usually 512
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    # Project back to input dimension
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def build_transformer_model(
    input_shape,
    num_classes=26,
    num_transformer_blocks=4,
    head_size=256,
    num_heads=8,
    ff_dim=512,
    mlp_units=[256, 128],
    dropout=0.2,
    mlp_dropout=0.2
):
    """
    Builds the Transformer model to match the specific 24-layer 
    weight-bearing structure in transformer_isl_*.h5.
    
    Architecture:
    1. Input Projection (Dense)
    2. N x Transformer Blocks (each with Norm, MHA, Norm, Conv, Conv)
    3. Global Average Pooling
    4. M x Dense layers (MLP head)
    5. Output Dense layer
    """
    inputs = tf.keras.Input(shape=input_shape)
    
    # 1. Input Projection Layer (corresponds to 'dense' in weight file)
    # Required to lift the 3D coordinates to the latent space for attention.
    x = layers.Dense(128)(inputs)
    
    # 2. Transformer Encoder Blocks (corresponds to 'multi_head_attention_*', 'conv1d_*', 'layer_normalization_*')
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    # 3. Dimensionality Reduction
    x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    
    # 4. MLP Classification Head (corresponds to 'dense_1', 'dense_2', 'dense_3')
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
        
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    return tf.keras.Model(inputs, outputs)
