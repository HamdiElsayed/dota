import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow_addons.layers import GroupNormalization
from tensorflow.keras.layers import TimeDistributed as td

class ConvBlock(layers.Layer):
    """ Convolutional block for downsampling and upsampling. used as building block for the encoder and decoder."""
    def __init__(self, num_channels=64, kernel_size=5, downsample=False,
        upsample=False, flatten=False):
        super(ConvBlock, self).__init__()
        
        # Build list of layers - conv, pooling, normalization, activation
        self.block = Sequential()
        self.block.add(layers.Conv3D(
            num_channels, kernel_size, kernel_regularizer=ws_reg, use_bias=False, padding='same'))
        self.block.add(layers.MaxPooling3D(pool_size=(1,2,2))) if downsample else None
        self.block.add(layers.Conv3DTranspose(
            num_channels, kernel_size, strides=(1,2,2), padding='same', use_bias=False)) if upsample else None
        self.block.add(layers.LayerNormalization())
        self.block.add(layers.LeakyReLU())
        self.block.add(layers.TimeDistributed(layers.Flatten('channels_last'))) if flatten else None
        
    def call(self, inputs):
        return self.block(inputs)


class TransformerEncoder(layers.Layer):
    """ Transformer encoder block."""
    def __init__(self,  num_heads, num_tokens, projection_dim, causal=True,
                dropout_rate=0.2, num_forward=0):
        super(TransformerEncoder, self).__init__()
        self.num_tokens = num_tokens
        self.num_heads = num_heads
        self.projection_dim = projection_dim
        
        # Multi-head self attention layer
        self.multihead = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.projection_dim,
            dropout=dropout_rate,
            kernel_initializer='truncated_normal',
            use_bias=False)
        
        # MLP stack layer
        self.mlp_network = Sequential([
            layers.Dense(self.projection_dim, activation=tf.nn.gelu),
            layers.Dropout(dropout_rate),
            layers.Dense(self.projection_dim, activation=tf.nn.gelu),
            layers.Dropout(dropout_rate)
            ])
        
        # Normalization layers
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.add = layers.Add()
        
        # Mask for causal attention
        if causal:
            self.mask = np.tri(num_tokens, num_tokens, num_forward, dtype=bool) 
        else:
            self.mask = np.ones((num_tokens, num_tokens))

    def call(self, tokens):
        x = self.norm1(tokens)
        x= self.multihead(x, x, attention_mask=self.mask ,return_attention_scores=False)
        x = self.add([x, tokens])
        y = self.norm2(x)
        y = self.mlp_network(y)
        return self.add([x,y]) 


class PosEmbedding(layers.Layer):
    """ Concatenate sequences and append position embedding."""
    def __init__(self, num_tokens, token_size):
        super(PosEmbedding, self).__init__()
        self.num_tokens = num_tokens
        self.token_size = token_size
        self.concat = layers.Concatenate(axis=1)
        self.embedding = layers.Embedding(
            input_dim=num_tokens, output_dim=token_size)
         # Linear transformation of the energy
        self.linear = layers.Dense(units=self.token_size)
        self.reshape = layers.Reshape((1, self.token_size))
        self.concat = layers.Concatenate(axis=1)

        self.embedding = layers.Embedding(
            input_dim=num_tokens, output_dim=token_size)

    def call(self,tokens,energies):

        
        e=self.reshape(self.linear(energies))
        x=self.concat([e,tokens])
        positions = tf.range(start=0, limit=self.num_tokens, delta=1)
        encoded= x+ self.embedding(positions)


        return encoded

def ws_reg(kernel):

    """ Function for weight standardization"."""
    kernel_mean = tf.math.reduce_mean(kernel, axis=[0, 1, 2], keepdims=True, name='kernel_mean')
    kernel = kernel - kernel_mean
    kernel_std = tf.keras.backend.std(kernel, axis=[0, 1, 2], keepdims=True)
    kernel = kernel / (kernel_std + 1e-5)



