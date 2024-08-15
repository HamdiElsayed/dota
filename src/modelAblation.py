
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
        num_heads (int): Number of attention heads in the multi-head attention mechanism. This is not used in the model.
        num_transformers (int): Number of transformer blocks. This is not used in the model.
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
    
    tokens = ConvBlock(enc_feats, kernel_size=kernel_size, flatten=True)(inputs)

    

    #Inform the tokens about the energy by multiplying the tokens with the energy.


    energies_expanded = tf.reshape(energies, shape=[-1, 1, 1])
    energies_expanded = tf.broadcast_to(energies_expanded, [tf.shape(tokens)[0], 150, 1])
    tokens = tf.multiply(tokens, energies_expanded)


    
    #Reshape the tokens back to the output shape of the convolutional encoder.
    x = layers.TimeDistributed(layers.Reshape((token_dim)))(tokens)
    # Upsample the tensor back to the original shape using the convolutional decoder
    for _ in range(steps):
        x = layers.Concatenate(axis=-1)([x, inputs_history.pop()])
        x = ConvBlock(enc_feats, kernel_size=kernel_size, upsample=True)(x)
    
    # Final convolution to produce the dose output
    dose = layers.Conv3D(1, kernel_size=kernel_size, padding='same')(x)

    return Model(inputs=[ct_vol, ray_vol, energies], outputs=dose)

