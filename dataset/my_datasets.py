import numpy as np
import torchvision.transforms.functional as tf
import random
from torch.utils import data
from torchvision import transforms
from PIL import Image
import torch


# ################ Dataset for Seg
class MyDataSet_seg(data.Dataset):
    def __init__(self, root_path, list_path, root_path_coarsemask=None, crop_size=(224, 224), max_iters=None):
        self.root_path = root_path
        self.root_path_coarsemask = root_path_coarsemask
        self.list_path = list_path
        self.crop_w, self.crop_h = crop_size

        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        for name in self.img_ids:
            img_file = name[0:name.find(';')]
            label_file = name[name.find(';')+1:]
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

        self.train_augmentation = transforms.Compose(
            [transforms.RandomAffine(degrees=10, translate=(0, 0.1), scale=(0.9, 1.1), shear=5.729578),
             transforms.RandomVerticalFlip(p=0.5),
             transforms.RandomHorizontalFlip(p=0.5),
             transforms.ToTensor(),
             transforms.ToPILImage(),
             transforms.Resize(self.crop_w)
             ])

        self.train_gt_augmentation = transforms.Compose(
            [transforms.RandomAffine(degrees=10, translate=(0, 0.1), scale=(0.9, 1.1), shear=5.729578),
             transforms.RandomVerticalFlip(p=0.5),
             transforms.RandomHorizontalFlip(p=0.5),
             transforms.ToTensor(),
             transforms.ToPILImage(),
             transforms.Resize(self.crop_w)
             ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(self.root_path + datafiles["img"])
        label = Image.open(self.root_path + datafiles["label"])

        is_crop = [0, 1]
        random.shuffle(is_crop)

        if is_crop[0] == 0:
            [WW, HH] = image.size
            p_center = [int(WW / 2), int(HH / 2)]
            crop_num = np.array(range(30, int(np.mean(p_center) / 2), 30))

            random.shuffle(crop_num)
            crop_p = crop_num[0]
            rectangle = (crop_p, crop_p, WW - crop_p, HH - crop_p)
            image = image.crop(rectangle)
            label = label.crop(rectangle)

            image = image.resize((self.crop_w, self.crop_h), Image.Resampling.BICUBIC)
            label = label.resize((self.crop_w, self.crop_h), Image.Resampling.NEAREST)

        else:
            image = image.resize((self.crop_w, self.crop_h), Image.Resampling.BICUBIC)
            label = label.resize((self.crop_w, self.crop_h), Image.Resampling.NEAREST)

        seed = np.random.randint(2147483647)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        image = self.train_augmentation(image)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        label = self.train_gt_augmentation(label)

        image = np.array(image) / 255.
        image = image.transpose((2, 0, 1))
        image = image.astype(np.float32)

        label = np.array(label)
        label = np.float32(label > 0)

        name = datafiles["img"][10:26]

        return image.copy(), label.copy(), name


class MyValDataSet_seg(data.Dataset):
    def __init__(self, root_path, list_path, root_path_coarsemask=None, crop_size=(224, 224)):
        self.root_path = root_path
        self.root_path_coarsemask = root_path_coarsemask
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.img_ids = [i_id.strip() for i_id in open(list_path)]

        self.files = []
        for name in self.img_ids:
            img_file = name[0:name.find(';')]
            label_file = name[name.find(';')+1:]
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(self.root_path + datafiles["img"])
        label = Image.open(self.root_path + datafiles["label"])

        image = image.resize((self.crop_h, self.crop_w), Image.Resampling.BICUBIC)
        label = label.resize((self.crop_h, self.crop_w), Image.Resampling.NEAREST)

        image = np.array(image) / 255.
        image = image.transpose(2, 0, 1)
        image = image.astype(np.float32)

        label = np.array(label)
        label = np.float32(label > 0)

        name = datafiles["img"][8:24]

        return image.copy(), label.copy(), name


class MyTestDataSet_seg(data.Dataset):
    def __init__(self, root_path, list_path, root_path_coarsemask=None, crop_size=(224, 224)):
        self.root_path = root_path
        self.root_path_coarsemask = root_path_coarsemask
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.img_ids = [i_id.strip() for i_id in open(list_path)]

        self.files = []
        for name in self.img_ids:
            img_file = name[0:name.find(' ')]
            label_file = name[name.find(' ')+1:]
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(self.root_path + datafiles["img"])
        label = Image.open(self.root_path + datafiles["label"])

        image0 = image.resize((self.crop_h, self.crop_w), Image.BICUBIC)
        image0 = np.array(image0) / 255.
        image0 = image0.transpose(2, 0, 1).astype(np.float32)

        image1 = image.resize((self.crop_h + 32, self.crop_w + 32), Image.BICUBIC)
        image1 = np.array(image1) / 255.
        image1 = image1.transpose(2, 0, 1).astype(np.float32)

        image2 = image.resize((self.crop_h + 64, self.crop_w + 64), Image.BICUBIC)
        image2 = np.array(image2) / 255.
        image2 = image2.transpose(2, 0, 1).astype(np.float32)

        label = np.array(label)

        name = datafiles["img"][7:23]

        return image0.copy(), image1.copy(), image2.copy(), label.copy(), name


# ################ Dataset for generating Coarsemask
class MyGenDataSet(data.Dataset):
    def __init__(self, root_path, list_path, mode=0, crop_size=(224, 224)):
        self.root_path = root_path
        self.list_path = list_path
        self.mode = mode
        self.crop_h, self.crop_w = crop_size
        self.img_ids = [i_id.strip() for i_id in open(list_path)]

        self.files = []
        for name in self.img_ids:
            img_file = name[0:name.find(' ')]
            self.files.append({
                "img": img_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image_ori = Image.open(self.root_path + datafiles["img"])
        if self.mode == 0:
            image = np.array(image_ori) / 255.
            image = image.transpose(2, 0, 1)
            image = image.astype(np.float32)
            name = datafiles["img"]
            return image.copy(), name
        else:
            image = image_ori.resize((self.crop_h, self.crop_w), Image.BICUBIC)
            image = np.array(image) / 255.
            image = image.transpose(2, 0, 1)
            image = image.astype(np.float32)
            image_ori = np.array(image_ori)
            name = datafiles["img"][7:23]
            return image_ori.copy(), image.copy(), name


if __name__ == "__main__":
    fold = 0
    data_train_root = '../../data/BUSI/'
    data_train_list = './ISIC/Training_seg_{}fold.txt'.format(str(0))
    trainloader = data.DataLoader(MyDataSet_seg(data_train_root, data_train_list, crop_size=(512, 512)),
                                  batch_size=4, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    data_val_root = '../../data/BUSI/'
    data_val_list = './ISIC/Validation_seg_{}fold.txt'.format(str(0))
    valloader = data.DataLoader(MyValDataSet_seg(data_val_root, data_val_list, crop_size=(512, 512)), batch_size=1, shuffle=False,
                                num_workers=4, pin_memory=True, drop_last=True)

    for i_iter, batch in enumerate(trainloader):
    # for i_iter, batch in enumerate(valloader):
        images, labels, name = batch
        print(i_iter, name, images.shape, labels.shape, images.max(), labels.max())
        # break





















