

import numpy as np
from src.Blocks import  ConvBlock, TransformerEncoder, PosEmbedding

import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import Tuple, Optional

def EDoTA(inshape: Tuple[int, ...], 
          steps: int, 
          enc_feats: int, 
          num_heads: int, 
          num_transformers: int,
          kernel_size: Tuple[int, int, int], 
          dropout_rate: float = 0.2, 
          causal: bool = False) -> Model:
    """
    Creates a transformer model for dose calculation in VHEE (Very High Energy Electron) radiation therapy.

    The model utilizes patient geometry, beam shape, and energy input to predict the radiation dose distribution.

    Args:
        inshape (Tuple[int, ...]): Shape of the input tensor.
        steps (int): Number of downsampling and upsampling blocks.
        enc_feats (int): Number of features in the encoder.
        num_heads (int): Number of attention heads in the multi-head attention mechanism.
        num_transformers (int): Number of transformer blocks.
        kernel_size (Tuple[int, int, int]): Size of the convolutional kernel.
        dropout_rate (float, optional): Dropout rate for regularization. Defaults to 0.2.
        causal (bool, optional): If True, uses causal attention. Defaults to False.

    Returns:
        Model: A Keras model for predicting radiation dose distribution.
    """

    slice_dim = inshape[1:]
    token_dim = (*[int(i / 2**steps) for i in slice_dim[:-1]], enc_feats)
    token_size = np.prod(token_dim)
    num_tokens = inshape[0] + 1

    # Input layers for CT values, ray volumes, and energies
    ct_vol = layers.Input(shape=(num_tokens - 1, *slice_dim))
    ray_vol = layers.Input(shape=(num_tokens - 1, *slice_dim))
    
    # Concatenate the CT and ray tensors
    inputs = layers.Concatenate(axis=-1)([ct_vol, ray_vol])
    
    # Input layer for energies
    energies = layers.Input(shape=(1,))

    inputs_history = [inputs]

    # Downsampling steps using convolutional blocks
    for _ in range(steps):
        inputs = ConvBlock(kernel_size=kernel_size, downsample=True)(inputs)
        inputs_history.append(inputs)
    
    # Flatten and encode the features into tokens
    tokens = ConvBlock(enc_feats, kernel_size=kernel_size, flatten=True)(inputs)
    
    # Add positional encoding to the tokens
    tokens = PosEmbedding(num_tokens, token_size)(tokens, energies)

    # Transformer encoder blocks, the multi-head attention mechanism is non-causal.
    for i in range(num_transformers):
        tokens = TransformerEncoder(
            num_heads=num_heads, 
            num_tokens=num_tokens, 
            projection_dim=token_size,
            causal=causal,
            dropout_rate=dropout_rate,
        )(tokens)

    # Reshape the tokens to the original dimensions, removing the energy token
    x = layers.TimeDistributed(layers.Reshape(token_dim))(tokens)
    x = tf.keras.layers.Lambda(lambda x: x[:, 1:, :, :, :])(x)
    
    # Upsample the tensor back to the original shape using the convolutional decoder
    for _ in range(steps):
        x = layers.Concatenate(axis=-1)([x, inputs_history.pop()])
        x = ConvBlock(enc_feats, kernel_size=kernel_size, upsample=True)(x)
    
    # Final convolution to produce the dose output
    dose = layers.Conv3D(1, kernel_size=kernel_size, padding='same')(x)

    return Model(inputs=[ct_vol, ray_vol, energies], outputs=dose)
