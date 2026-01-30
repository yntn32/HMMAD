import torch
import torch.nn as nn

from Net import NET_MRI, NET_PET
from Net.crosstransformer import CrossTransformer_MOD_AVG


class NET_DUAL(nn.Module):
    def __init__(self, class_num, input_shape, patch_size):
        super().__init__()

        self.mri_network = NET_MRI.NET_MRI(class_num, input_shape, patch_size)
        self.pet_network = NET_PET.NET_PET(class_num, input_shape, patch_size)

        self.CrossAttentionLeft = CrossTransformer_MOD_AVG(128, 4, 8, 64, 512, 0.1)
        self.CrossAttentionRight = CrossTransformer_MOD_AVG(128, 4, 8, 64, 512, 0.1)

        # # classification

        self.classifier = nn.Sequential(
            nn.Linear(2048, 32),
            nn.ReLU(True),
            nn.Linear(32, class_num),
            nn.Softmax(dim=1)
        )

    def forward(self, x_mri, x_pet):
        b = x_mri.shape[0]
        P_xl1, P_xr1, PA_xl1, PA_xr1, PAC_xl1, PAC_xr1, PAG_cls1 = self.mri_network(x_mri)
        P_xl2, P_xr2, PA_xl2, PA_xr2, PAC_xl2, PAC_xr2, PAG_cls2 = self.pet_network(x_pet)

        CLS_1, PA_xl1, PA_xl2 = self.CrossAttentionLeft(PA_xl1, PA_xl2)
        CLS_2, PA_xr1, PA_xr2 = self.CrossAttentionRight(PA_xr1, PA_xr2)

        cls = torch.cat((CLS_1, CLS_2, PAG_cls1, PAG_cls2), dim=1)
        cls = self.classifier(cls)

        return cls


if __name__ == '__main__':
    x1 = torch.randn(3, 1, 105, 125, 105)
    x2 = torch.randn(3, 1, 105, 125, 105)
    model = NET_DUAL(2, input_shape=[105, 125, 105], patch_size=25)
    output = model(x1, x2)
    print(len(output))
    print(output)
