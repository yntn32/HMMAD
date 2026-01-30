"""

Remove the periphery while keeping the core in pieces. Function The input tensor is adjusted through the "remove
edges and keep core" method to a form that can be precisely cropped by the patch_size, and then it is cut into a
group of small pieces.

"""
import torch
from einops import rearrange
from torch import nn


class Patches_CropSelected(nn.Module):  # for brain neuroimaging (edge patch will be excluded)
    def __init__(self, input_shape, patch_size, crop_mode="center"):
        super().__init__()

        assert (len(input_shape) == 3)
        assert (isinstance(patch_size, int) or (isinstance(patch_size, list) and len(patch_size) == 3))

        if isinstance(patch_size, int):
            self.patch_size = [patch_size, patch_size, patch_size]
        else:
            self.patch_size = patch_size

        self.new_ipt_shape = [input_shape[idx] % self.patch_size[idx] for idx in range(3)]
        self.edge1_shape = [self.new_ipt_shape[idx] // 2 for idx in range(3)]
        self.edge2_shape = [self.new_ipt_shape[idx] - self.edge1_shape[idx] for idx in range(3)]

        for i in range(3):
            if self.edge2_shape[i] == 0:
                self.edge2_shape[i] = None
            else:
                self.edge2_shape[i] = -self.edge2_shape[i]
        self.patches_shape = torch.tensor(input_shape) // self.patch_size[0]

    def forward(self, x):
        B = x.shape[0]

        x = x[:, :,
            self.edge1_shape[0]:self.edge2_shape[0],
            self.edge1_shape[1]:self.edge2_shape[1],
            self.edge1_shape[2]:self.edge2_shape[2]]

        patchs_output = rearrange(x, 'b c (h1 ph) (w1 pw) (d1 pd) -> (b h1 w1 d1) c ph pw pd',
                                  ph=self.patch_size[0], pw=self.patch_size[1], pd=self.patch_size[2])

        patchs_output = rearrange(patchs_output, '(b p) ... -> b p ...', b=B)

        patchs_output = rearrange(patchs_output, 'b p c ... -> b (p c) ...', b=B)

        return patchs_output


if __name__ == '__main__':
    model = Patches_CropSelected([50, 125, 100], 25)
    x = torch.randn((3, 1, 50, 125, 100))
    y = model(x)
    print(y.shape)
    # torch.Size([3, 40, 25, 25, 25])
    print(model.patches_shape)
    # tensor([2, 5, 4])
