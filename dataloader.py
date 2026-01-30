import gc
import nibabel
from torch.utils.data import Dataset, DataLoader
import os
import torch
from configp import get_args
import numpy as np

def norm_img(img):
    img = img.astype('float32')
    normalization = 'minmax'
    if normalization == 'minmax':
        img_max = img.max()
        img = img / img_max
    elif normalization == 'median':
        img_fla = np.array(img).flatten()
        index = np.argwhere(img_fla == 0)
        img_median = np.median(np.delete(img_fla, index))
        img = img / img_median

    if normalization == 'minmax':
        del img_max
    else:
        del img_fla, index, img_median
    gc.collect()

    return img

class DataSet(Dataset):
    def __init__(self, MRI_path, PET_path, class_dir):
        self.MRI_path = MRI_path
        self.PET_path = PET_path
        self.class_dir = class_dir
        self.MRI_image_path = os.path.join(self.MRI_path, class_dir)
        self.PET_image_path = os.path.join(self.PET_path, class_dir)
        self.MRI_images = os.listdir(self.MRI_image_path)
        self.PET_images = os.listdir(self.PET_image_path)

    def __getitem__(self, index):
        label = 0
        MRI_image_index = self.MRI_images[index]
        PET_image_index = self.PET_images[index]
        MRI_img_path = os.path.join(self.MRI_image_path, MRI_image_index)  # 获取数据的路径或目录
        PET_img_path = os.path.join(self.PET_image_path, PET_image_index)  # 获取数据的路径或目录
        MRI_img = nibabel.load(MRI_img_path).get_fdata()  # 读取数据
        PET_img = nibabel.load(PET_img_path).get_fdata()  # 读取数据

        MRI_img = norm_img(MRI_img)
        PET_img = norm_img(PET_img)
        MRI_img = np.expand_dims(MRI_img, axis=0)
        PET_img = np.expand_dims(PET_img, axis=0)

        # labeling AD / NC
        if self.class_dir == 'AD/':
            label = label+1
        elif self.class_dir == 'CN/':
            label = label
        elif self.class_dir == 'MCI/':
            label = label+2

        return MRI_img, PET_img, label

    def __len__(self):
        return len(self.MRI_images)

def load_data(args, MRI_path, PET_path, AD_path, CN_path):
    train_AD = DataSet(MRI_path, PET_path, AD_path)
    train_CN = DataSet(MRI_path, PET_path, CN_path)
    trainDataset = train_AD + train_CN
    train_loader = DataLoader(trainDataset, batch_size=args.batch_size, shuffle=True)
    del trainDataset
    gc.collect()
    return train_loader


# args = get_args()
# train_data, test_data = load_data(args)
# for step, (b_x, b_y) in enumerate(train_data):
#     if step > 1:
#         break
#
# print(b_x.shape)
# print(b_y.shape)
# print(b_x)
# print(b_y)