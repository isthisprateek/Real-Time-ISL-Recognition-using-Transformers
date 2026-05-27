import tensorflow as tf
from tensorflow.keras import layers

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def build_transformer_model(
    input_shape,
    head_size=256,
    num_heads=4,
    ff_dim=256,
    num_transformer_blocks=4,
    mlp_units=[128],
    dropout=0,
    mlp_dropout=0,
    num_classes=26
):
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    
    # Optional: Initial Dense projection to higher dimensionality
    x = layers.Dense(128)(x)
    
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
        
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model

if __name__ == "__main__":
    # Test building the model
    # input_shape = (21, 3) # 21 landmarks, each with x, y, z
    model = build_transformer_model(input_shape=(21, 3), num_classes=26)
    model.summary()
