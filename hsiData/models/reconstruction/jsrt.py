import numpy as np
import math
import warnings
from einops import rearrange
import flax.linen as nn 
from typing import Any, Callable, Sequence

import jax 
import jax.numpy as jnp 
import torch



class SpectralMultiHeadAttention(nn.Module):
    spectra_channels : int   
    num_heads : int  

    def setup(self):
        self.wQ = nn.Dense(
            features = self.spectra_channels,
            kernel_init = nn.initializers.kaiming_normal(),
            use_bias = False)
        self.wK = nn.Dense(
            features = self.spectra_channels,
            kernel_init = nn.initializers.kaiming_normal(),
            use_bias = False)
        self.wV = nn.Dense(
            features = self.spectra_channels,
            kernel_init = nn.initializers.kaiming_normal(),
            use_bias = False)

        self.projOUT = nn.Dense(
            features = self.spectra_channels,
            kernel_init = nn.initializers.kaiming_normal(),
            use_bias = False)

    def split_heads(self, x):
        """
        batch, dims = hw, seq = c
        to
        batch, heads, seq = c, dims = hw
        """
        # x >> b, hw, c
        x = x.transpose(0, 2, 1)                  # b, c, hw

        # n >> hw, 
        x = rearrange(x, 'b (heads c) hw -> b heads c hw', heads=self.num_heads)
        return x

    def normalize_last(self, x):
        norm = jnp.linalg.norm(x, axis = -1)
        return x/norm[..., None]

    def __call__(self, x, mask = None):
        b, h, w, c = x.shape
        x = x.reshape(b, h*w, c)   # consider c the sequence

        q = self.wQ(x)             # batch, dims = hw, seq = c
        k = self.wK(x)
        v = self.wV(x) 

        # separate the heads
        q = self.split_heads(q)     # b, heads, c/heads, hw
        k = self.split_heads(k)  
        v = self.split_heads(v)  

        
        q = self.normalize_last(q)
        k = self.normalize_last(k)

        # # compute the value outputs
        values, attn_score = self.scaledDotProduct(q, k, v, mask = mask)

        # # values >> # batch, head, seq, dims
        values = values.transpose(0, 3, 1, 2)   # batch, dims = hw/head, head, seq = c
        values = values.reshape(b, h*w, c)   # batch, dims = hw/head, head, seq = c

        output = self.projOUT(values)
        output = output.reshape(b, h, w, c)     # batch, seq = c, dims = hw
        return output   

    def scaledDotProduct(self, q, k, v, mask=None):
        d_k = q.shape[-1]
        qk = jnp.matmul(q, jnp.swapaxes(k, -2, -1))
        attn = qk / math.sqrt(d_k)
        if mask is not None:
            attn = jnp.where(mask == 0, -9e15, attn)

        attn_score = nn.softmax(attn, axis=-1)
        values = jnp.matmul(attn_score, v)
        return values, attn_score

class GConv2d(nn.Module):
    out_channels : int   

    def setup(self):
        self.CV = nn.Conv(
                features = self.out_channels,
                kernel_size = (1, 1), 
                strides = 1, 
                use_bias = False)

    def __call__(self, x):
        x = self.CV(x)
        x = nn.gelu(x)
        return x
            
class NormFFN(nn.Module):
    expected_channels : int 
    factor : int = 4

    def setup(self):
        self.NORM = nn.LayerNorm()
        self.GFF = nn.Sequential([
          GConv2d(out_channels = self.expected_channels * self.factor),
          nn.Conv(
                features = self.expected_channels * self.factor,
                kernel_size = (3, 3), 
                strides = 1, 
                padding = 'same',
                feature_group_count = self.expected_channels * self.factor,
                use_bias = False),
          nn.gelu,
          nn.Conv(
                features = self.expected_channels,
                kernel_size = (1, 1), 
                strides = 1, 
                padding = 'same',
                use_bias = False)
        ])
            
    def __call__(self, x):
        x = self.NORM(x)               # b, h, w, c
        out = self.GFF(x)              # b, h, w, c
        return out

class AttnBlock(nn.Module):
    expected_channels : int = 31
    num_heads : int = 1
    num_blocks : int = 1

    def setup(self):
        self.blocks = [[
            SpectralMultiHeadAttention(
                    spectra_channels = self.expected_channels, 
                    num_heads = self.num_heads),
            NormFFN(
                    expected_channels = self.expected_channels,
                    factor = 4)
        ] for _ in range(self.num_blocks)]

    def __call__(self, x):
        for attn, nff in self.blocks:
            x = attn(x) + x
            x = nff(x) + x
        return x

class SpectraEncoder(nn.Module):
    spectra_channels : int = 31
    expected_channels : int = 31
    num_blocks : int = 1

    def setup(self):
        self.EB = AttnBlock(
            expected_channels = self.expected_channels, 
            num_blocks = self.num_blocks, 
            num_heads = self.expected_channels // self.spectra_channels)
        self.ED = nn.Conv(
            features = self.expected_channels * 2, 
            kernel_size = (4, 4), 
            strides = 2, 
            padding = 'same', 
            use_bias = False)

    def __call__(self, x):
        x_enc = self.EB(x)
        x_down = self.ED(x_enc)
        return x_enc, x_down

class SpectraDecoder(nn.Module):
    spectra_channels : int = 31
    expected_channels : int = 31
    num_blocks : int = 1

    def setup(self):
        self.DU = nn.ConvTranspose(
                    features = self.expected_channels // 2, 
                    strides = (2, 2), 
                    kernel_size = (2, 2), 
                    padding = 'VALID')
        self.DC = nn.Conv(
                    features = self.expected_channels // 2, 
                    strides = 1, 
                    kernel_size = (1, 1), 
                    use_bias = False)
        self.DB = AttnBlock(
                    expected_channels = self.expected_channels // 2, 
                    num_blocks = self.num_blocks, 
                    num_heads = (self.expected_channels // 2) // self.spectra_channels)

    def __call__(self, x, x_enc):
        x_up = self.DU(x)        
        x_fusion = self.DC(jnp.concatenate((x_up, x_enc), axis = -1))
        x_out = self.DB(x_fusion)
        return x_out 

class TransformerBlock(nn.Module):
    spectra_channels : int = 31
    num_blocks : Sequence[int] = jnp.array([1,1])

    def setup(self):
        self.len_blocks = len(self.num_blocks)

        # spatial projection
        self.SE1 = nn.Conv(
            features = self.spectra_channels, 
            kernel_size = (3, 3), 
            strides = 1, 
            padding = 'same', 
            use_bias = False)

        # Encoder
        expected_channels = self.spectra_channels
        temp = []
        for nb in self.num_blocks:
            temp.append(SpectraEncoder(
                spectra_channels = self.spectra_channels,
                expected_channels = expected_channels,
                num_blocks = nb
            ))
            expected_channels *= 2
        self.EN = temp

        # Intermediate
        self.INT = AttnBlock(
            expected_channels = expected_channels, 
            num_heads = expected_channels // self.spectra_channels, 
            num_blocks = self.num_blocks[-1])

        # Decoder
        temp = []
        for i, nb in enumerate(self.num_blocks[::-1]):
            temp.append(SpectraDecoder(
                spectra_channels = self.spectra_channels,
                expected_channels = expected_channels,
                num_blocks = nb
            ))
            expected_channels //= 2
        self.DC = temp

        # spatial projection
        self.SE2 = nn.Conv(
            features = self.spectra_channels, 
            kernel_size = (3, 3), 
            strides = 1, 
            padding = 'same', 
            use_bias = False)



    def __call__(self, x):
        # extract spatial details
        out = self.SE1(x)

        # Encoder
        fea_enc = []
        for ENC in self.EN:
            x_enc, out = ENC(out)
            fea_enc.append(x_enc)
            
        out = self.INT(out)
        
        # Decoder
        for i, DEC in enumerate(self.DC):
            out = DEC(out, fea_enc[self.len_blocks-1-i])

        # spatial projection
        out = self.SE2(out) + x

        return out

class SpectraUp(nn.Module):
    queries : Sequence[int] = jnp.array([700, 550, 450], dtype = jnp.float32)
    keys : Sequence[int] = jnp.arange(400, 710, 10, dtype = jnp.float32)
    option : int = 2
    do_softmax : bool = False

    max_band = jnp.max(jnp.array([jnp.max(queries), jnp.max(keys)]))
    
    def setup(self):
        self.w_factor = nn.Dense(
              features = self.keys.shape[0],
              kernel_init = nn.initializers.ones,
              use_bias = False
        )

    def computeDist(self, option = 1, do_softmax = False):
        if option == 1:  # subtract the differences
            dist = 1- ((self.queries[..., None] - self.keys)/self.max_band)**2
        elif option == 2:  # kernel distances
            dist = ((self.queries[..., None] - self.keys)/self.max_band)**2
            var = jnp.sum(dist, axis = -1)/len(self.keys)
            std = jnp.sqrt(var)
            dist = 1/(jnp.sqrt((2*math.pi))*std[..., None])*jnp.exp(-(dist/(2*var[..., None])))
        
        if do_softmax:
            dist = nn.softmax(dist, axis = -1)
        return dist

    def __call__(self, values):
        dist = self.computeDist(option=self.option, do_softmax=self.do_softmax)
        
        weight = self.w_factor(dist)
        outputs = values @ weight
        return outputs
        # pass 


class model(nn.Module):
    cascade : int = 3
    rgb_bands = jnp.array([700, 550, 450], dtype = jnp.float32)
    spectra_bands = jnp.arange(400, 710, 10, dtype = jnp.float32)

    def setup(self):
        spectra_channels = len(self.spectra_bands)
        rgb_channels = len(self.rgb_bands)

        # UP
        self.SU = SpectraUp(
                        queries = self.rgb_bands, 
                        keys = self.spectra_bands, 
                        option = 2, 
                        do_softmax = False)


        self.CU = nn.Conv(
                features = spectra_channels, 
                kernel_size = (3, 3), 
                strides = 1, 
                padding = 'same', 
                use_bias = False)

        # # # Early Fusion
        self.EF_w1 = nn.Dense(
            features = spectra_channels,
            kernel_init = nn.initializers.kaiming_normal(),
            use_bias = False)
        self.EF_w2 = nn.Dense(
            features = spectra_channels,
            kernel_init = nn.initializers.kaiming_normal(),
            use_bias = False)
        self.EF_CN =  nn.Sequential([
            GConv2d(
                out_channels = spectra_channels * 3),
            nn.Conv(
                features = spectra_channels, 
                kernel_size = (3, 3), 
                strides = 1, 
                padding = 'same',
                feature_group_count = spectra_channels, 
                use_bias = False)])

        CTB = [TransformerBlock(
            spectra_channels = spectra_channels, 
            num_blocks = [1, 1]) for _ in range(self.cascade)]
        self.BO = nn.Sequential(CTB)


        self.CO = nn.Conv(
                features = spectra_channels, 
                kernel_size = (3, 3), 
                strides = 1, 
                padding = 'same', 
                use_bias = False)
        



    def __call__(self, x):
        b, h, w, c = x.shape
        x = padding(x, h, w)
        b, hp, wp, c = x.shape

        # Up
        x_s = self.SU(x)   
        x_c = self.CU(x)  
   
        # # # Early fusion
        hsi_s = self.EF_w1(x_s)
        hsi_s = hsi_s + x_s
  

        hsi_c = self.EF_w2(x_c)
        hsi_c = hsi_c + x_c

        x = self.EF_CN(hsi_s + hsi_c)

        hsi_out = self.BO(x) 
        hsi_out = self.CO(hsi_out) + x


             
        return hsi_out[:, :h, :w, :]





def padding(x, h_inp, w_inp):
    hb, wb = 8, 8
    pad_h = (hb - h_inp % hb) % hb
    pad_w = (wb - w_inp % wb) % wb
    x = jnp.pad(x, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]], mode='reflect')
    return x

def shift(rgb):
    batch, c, h, w = rgb.shape
    x_mean = torch.mean(rgb, axis = (2,3))
    x_std = torch.std(rgb, axis = (2,3))

    a = torch.sqrt(1/x_std**2)
    b = 0 - a * x_mean

    rgb_n = rgb * a[..., None, None].expand(batch, c, h, w) + b[..., None, None].expand(batch, c, h, w)

    # rgb_n = (torch.nn.Tanh()(rgb_n)+1)/2
    # rgb_nn = torch.zeros_like(rgb_n)
    # for k in range(len(rgb_n)):
    #     rgb_nn[k] = (rgb_n[k] - torch.min(rgb_n[k])) / (torch.max(rgb_n[k]) - torch.min(rgb_n[k]))
    # rgb_n = (rgb_n - torch.min(rgb_n)) / (torch.max(rgb_n) - torch.min(rgb_n))

    temp = rgb_n.reshape(b, c*h*w)
    minvals = torch.min(temp, dim=1).values
    maxvals = torch.max(temp, dim=1).values 

    return rgb_n












