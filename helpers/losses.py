from __future__ import division

import torch
import numpy as np
import cv2
import flax.linen as nn 
from typing import Any, Callable, Sequence, Optional, Tuple
import chex
import jax.numpy as jnp 
import optax



def MAE(
        pred: chex.Array,
        true: Optional[chex.Array] = None,
    ) -> chex.Array:
    chex.assert_type([pred], float)
    assert pred.shape == true.shape
    error = jnp.abs(pred - true) 
    mae = jnp.mean(error.reshape(-1))
    return mae

def MRAE(
        pred: chex.Array,
        true: Optional[chex.Array] = None,
    ) -> chex.Array:
    chex.assert_type([pred], float)
    assert pred.shape == true.shape
    error = jnp.abs(pred - true) / true
    mrae = jnp.mean(error.reshape(-1))
    return mrae

def RMSE(
        pred: chex.Array,
        true: Optional[chex.Array] = None,
    ) -> chex.Array:
    chex.assert_type([pred], float)
    assert pred.shape == true.shape
    error = pred-true
    sqrt_error = jnp.pow(error,2)
    rmse = jnp.sqrt(jnp.mean(sqrt_error.reshape(-1)))
    return rmse

def PSNR(
        pred: chex.Array,
        true: Optional[chex.Array] = None,
        data_range: int = 255
    ) -> chex.Array:
    chex.assert_type([pred], float)

    pred = pred * data_range
    true = true * data_range
    mse = optax.l2_loss
    err = mse(pred, true).mean()
    psnr = 10. * jnp.log10((data_range ** 2) / err) 
    return jnp.mean(psnr)


def SSIM(img1, img2):
       C1 = (0.01)**2
       C2 = (0.03)**2

       img1 = np.asarray(img1[0])
       img2 = np.asarray(img2[0])
       kernel = cv2.getGaussianKernel(11, 1.5)
       window = np.outer(kernel, kernel.transpose())

       mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
       mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
       mu1_sq = mu1**2
       mu2_sq = mu2**2
       mu1_mu2 = mu1 * mu2
       sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
       sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
       sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

       ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                               (sigma1_sq + sigma2_sq + C2))
       return ssim_map.mean()