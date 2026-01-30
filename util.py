import os

import torch
from torch import nn


def model_save(model: nn.Module, path):
    print('Saving model...')
    dir_path_temp = os.path.dirname(path)
    if not os.path.exists(dir_path_temp):
        os.makedirs(dir_path_temp)
    torch.save(model.state_dict(), path)


def image_reshape(src, new_shape):
    new_shape = [src.shape[0], src.shape[1]] + new_shape
    zero = torch.zeros(new_shape)

    margin_x = new_shape[2] - src.shape[2]
    margin_y = new_shape[3] - src.shape[3]
    margin_z = new_shape[4] - src.shape[4]

    margin_x_1 = margin_x // 2
    margin_y_1 = margin_y // 2
    margin_z_1 = margin_z // 2

    margin_x_2 = margin_x - margin_x_1
    margin_y_2 = margin_y - margin_y_1
    margin_z_2 = margin_z - margin_z_1

    if margin_x_1 == 0:
        margin_x_1 = None
    if margin_y_1 == 0:
        margin_y_1 = None
    if margin_z_1 == 0:
        margin_z_1 = None

    if margin_x_2 == 0:
        margin_x_2 = None
    else:
        margin_x_2 = -margin_x_2

    if margin_y_2 == 0:
        margin_y_2 = None
    else:
        margin_y_2 = -margin_y_2

    if margin_z_2 == 0:
        margin_z_2 = None
    else:
        margin_z_2 = -margin_z_2

    zero[:, :,margin_x_1:margin_x_2, margin_y_1:margin_y_2, margin_z_1:margin_z_2] = src[:, :, :, :, :]
    return zero


import torch
import math


def positional_encoding(seq_len, d_model):
    position = torch.arange(seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe
