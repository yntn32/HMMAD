import torch
from einops import rearrange
from torch import nn

import Net.attention
from Net.Patch_Blocking import Patches_CropSelected
from Net.PatchFeatureExtractor import BaseNet
from Net.attention import VisionTransformerBlock
import torch.nn.functional as F
from Net.crosstransformer import CrossTransformer_MOD_AVG

class NET_PET(nn.Module):
    def __init__(self, class_num, input_shape, patch_size):
        super().__init__()
        self.input_shape = input_shape

        self.input_cut = [(input_shape[i] % patch_size) for i in range(3)]
        self.input_cut1 = [self.input_cut[i] // 2 for i in range(3)]
        self.input_cut2 = [self.input_cut[i] - self.input_cut1[i] for i in range(3)]
        for i in range(3):
            if self.input_cut2[i] == 0:
                self.input_cut2[i] = None
            else:
                self.input_cut2[i] = -self.input_cut2[i]

        input1_shape = [input_shape[i] - self.input_cut[i] for i in range(3)]
        input2_shape = [input_shape[i] - self.input_cut[i] for i in range(3)]
        input1_shape[0] = input1_shape[0] // 2
        input2_shape[0] = input2_shape[0] // 2

        # The segmentation function takes the size of the image to be segmented and the size of the blocks as input,
        # and outputs the segmentation result.
        self.input1 = Patches_CropSelected(input_shape=input1_shape, patch_size=patch_size)  # 去边留芯分块块
        self.input2 = Patches_CropSelected(input_shape=input2_shape, patch_size=patch_size)

        # Extract features from the patch, with the input being the number of channels of the four convolutional layers.
        self.patchFeatureExtractorLeft = BaseNet([32, 64, 128, 128])
        self.patchFeatureExtractorRight = BaseNet([32, 64, 128, 128])

        # Calculate how many sections the left brain has been divided into,
        # and how many sections the right brain has been divided into.
        self.LeftPatchNumber = torch.prod(self.input1.patches_shape)
        self.RightPatchNumber = torch.prod(self.input2.patches_shape)

        self.attentionLeft = VisionTransformerBlock(128, 8)
        self.attentionRight = VisionTransformerBlock(128, 8)
        self.crossAttention = CrossTransformer_MOD_AVG(128, 4, 8, 64, 512, 0.1)

        self.classifierLeft = nn.Sequential(
            nn.Linear(torch.prod(torch.tensor([self.LeftPatchNumber, 128])), 32),
            nn.ReLU(True),
            nn.Linear(32, class_num),
            nn.Softmax(dim=1),
        )
        self.classifierRight = nn.Sequential(
            nn.Linear(torch.prod(torch.tensor([self.LeftPatchNumber, 128])), 32),
            nn.ReLU(True),
            nn.Linear(32, class_num),
            nn.Softmax(dim=1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 32),
            nn.ReLU(True),
            nn.Linear(32, class_num),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        b = x.shape[0]  # batch number

        x = x[:, :, self.input_cut1[0]:self.input_cut2[0],
            self.input_cut1[1]:self.input_cut2[1],
            self.input_cut1[2]:self.input_cut2[2]]

        # Dividing the left and right brains
        x1 = x[:, :, :(x.shape[2] // 2), :, :]  # left
        x2 = x[:, :, (x.shape[2] // 2):, :, :]  # right
        x2 = torch.flip(x2, dims=[2])

        # In patch
        x1 = self.input1(x1)
        x2 = self.input2(x2)
        # x1.shape == x2.shape == [b, num, patch_h, patch_w, patch_d]

        x1 = rearrange(x1, 'b num (c h) w d -> (b num) c h w d', c=1)
        x2 = rearrange(x2, 'b num (c h) w d -> (b num) c h w d', c=1)

        # Patch feature extraction
        x1, x1_score = self.patchFeatureExtractorLeft(x1)
        x2, x2_score = self.patchFeatureExtractorRight(x2)
        # x1.shape :: [b*num, c, h, w, d]
        # if patch==25:
        #   x1.shape == [b*num, 128, 6, 6, 6]

        x1 = F.adaptive_avg_pool3d(x1, (1, 1, 1))
        x2 = F.adaptive_avg_pool3d(x2, (1, 1, 1))
        # x1.shape :: [b*40, 128, 1, 1, 1]

        x1 = rearrange(x1, '(b num) c h w d -> b num (c h w d)', b=b)
        x2 = rearrange(x2, '(b num) c h w d -> b num (c h w d)', b=b)
        # x.shape :: [b, num, 128]

        left_features_flat = x1
        right_features_flat = x2

        x1 = self.attentionLeft(x1)
        x2 = self.attentionRight(x2)
        cls, x1G, x2G = self.crossAttention(x1, x2)


        return left_features_flat, right_features_flat, x1, x2, x1G, x2G, cls


if __name__ == '__main__':
    x = torch.randn(3, 1, 105, 125, 105)
    model = NET_PET(2, input_shape=[105, 125, 105], patch_size=25)
    output = model(x)
    print(output)
