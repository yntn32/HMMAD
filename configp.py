import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--class_num', type=int, help='class_num', default=2)
    parser.add_argument('--seed', type=int, help='Seed', default=42)
    parser.add_argument('--gpu', type=str, help='GPU ID', default='0')
    parser.add_argument('--train_root_path_PET', type=str, help='Root path for train dataset',
                        default='D:/PET_MRI/dataset/dataset14/petmkfiv14/train/')
    parser.add_argument('--test_root_path_PET', type=str, help='Root path for test dataset',
                        default='D:/PET_MRI/dataset/dataset14/petmkfiv14/test/')
    parser.add_argument('--val_root_path_PET', type=str, help='Root path for val dataset',
                        default='D:/PET_MRI/dataset/dataset14/petmkfiv14/val/')
    parser.add_argument('--train_root_path_MRI', type=str, help='Root path for train dataset',
                        default='D:/PET_MRI/dataset/dataset14/mribrfiv14/train/')
    parser.add_argument('--test_root_path_MRI', type=str, help='Root path for test dataset',
                        default='D:/PET_MRI/dataset/dataset14/mribrfiv14/test/')
    parser.add_argument('--val_root_path_MRI', type=str, help='Root path for val dataset',
                        default='D:/PET_MRI/dataset/dataset14/mribrfiv14/val/')
    parser.add_argument('--batch_size', type=int, help='batch_size of data', default=6)
    parser.add_argument('--nepoch', type=int, help='Total epoch num', default=80)
    return parser.parse_args()
