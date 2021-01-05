import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
from torchvision import transforms as T
import torchvision
import cv2
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from config import Config
import random
import warnings
warnings.filterwarnings("error", category=UserWarning)

class rubbishDataset(Dataset):

    def __init__(self, root, data_list_file, phase='train', input_size=640):
        self.phase = phase

        with open(os.path.join(data_list_file), 'r') as fd:
            imgs = fd.readlines()

        imgs = [os.path.join(root, img.strip('\n')) for img in imgs]
        self.imgs = np.random.permutation(imgs)

        normalize = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


        if self.phase == 'train':
            self.transforms = T.Compose([
                T.Resize((int(input_size / 0.934), int(input_size / 0.934))),
                T.RandomCrop((input_size,input_size)),
                T.RandomRotation(10),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])
        else:
            self.transforms = T.Compose([
                T.Resize((int(input_size / 0.934), int(input_size / 0.934))),
                T.CenterCrop((input_size,input_size)),
                T.ToTensor(),
                normalize
            ])

    def __getitem__(self, index):
        sample = self.imgs[index]
        splits = sample.split(',')
        img_path = splits[0]
        # try:
        data = Image.open(img_path)
        # except:
        #     print(index)
        #     print(sample)
        #     print(img_path)
        data = data.convert('RGB')
        data = self.transforms(data)
        label = np.int32(splits[1].strip(' '))
        return data.float(), label


    def __len__(self):
        return len(self.imgs)

class w_rubbishDataset(Dataset):
    def __init__(self, root, data_list_file, phase='train', input_size=640,sample='balance'):
        self.phase = phase
        self.sample=sample

        with open(os.path.join(data_list_file), 'r') as fd:
            imgs = fd.readlines()

        imgs = [os.path.join(root, img.strip('\n')) for img in imgs]
        self.imgs = np.random.permutation(imgs)
        self.class_dict=self._get_class_dict()
        self.num_classes=43

        normalize = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

        if self.sample=='reverse':
            self.class_weight, self.sum_weight = self.get_weight(self.imgs, self.num_classes)


        if self.phase == 'train':
            self.transforms = T.Compose([
                T.Resize((int(input_size / 0.934), int(input_size / 0.934))),
                # T.RandomAffine(degrees=10,translate=(0.1,0.1)),
                # T.ColorJitter(brightness=0.1,hue=0.1,contrast=0.15,saturation=0.1),
                T.RandomCrop((input_size,input_size)),
                T.RandomRotation(10),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])
        else:
            self.transforms = T.Compose([
                T.Resize((int(input_size / 0.934), int(input_size / 0.934))),
                T.CenterCrop((input_size,input_size)),
                T.ToTensor(),
                normalize
            ])

    def get_weight(self, annotations, num_classes):
        num_list = [0] * num_classes
        cat_list = []
        for anno in annotations:
            category_id = int(anno.split(' ')[-1])
            num_list[category_id] += 1
            cat_list.append(category_id)
        max_num = max(num_list)
        class_weight = [max_num / i for i in num_list]
        sum_weight = sum(class_weight)
        return class_weight, sum_weight

    def sample_class_index_by_weight(self):
        rand_number, now_sum = random.random() * self.sum_weight, 0
        for i in range(self.num_classes):
            now_sum += self.class_weight[i]
            if rand_number <= now_sum:
                return i


    def _get_class_dict(self):
        class_dict = dict()
        for i, anno in enumerate(self.imgs):
            cat_id = (
                int(anno.split(' ')[-1])
            )
            if not cat_id in class_dict:
                class_dict[cat_id] = []
            class_dict[cat_id].append(i)
        return class_dict

    def __getitem__(self, index):
        if self.sample=='reverse':
            sample_class = self.sample_class_index_by_weight()
        else:
            sample_class = random.randint(0, 43 - 1)
        sample_indexes = self.class_dict[sample_class]
        index = random.choice(sample_indexes)

        sample = self.imgs[index]
        splits = sample.split(',')
        img_path = splits[0]
        # try:
        data = Image.open(img_path)
        # except:
        #     print(index)
        #     print(sample)
        #     print(img_path)
        data = data.convert('RGB')
        data = self.transforms(data)
        label = np.int32(splits[1].strip(' '))
        return data.float(), label


    def __len__(self):
        return len(self.imgs)






class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = T.ToTensor()

    def __call__(self, img):
        # padding
        ratio = self.size[0] / self.size[1]
        w, h = img.size
        if w / h < ratio:
            t = int(h * ratio)
            w_padding = (t - w) // 2
            img = img.crop((-w_padding, 0, w+w_padding, h))
        else:
            t = int(w / ratio)
            h_padding = (t - h) // 2
            img = img.crop((0, -h_padding, w, h+h_padding))

        # img.show()
        # resize
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img



if __name__ == '__main__':
    opt=Config()
    dataset = w_rubbishDataset(root='../70',
                      data_list_file='../dataset/train.txt',
                      phase='test',
                      input_size=opt.input_size,sample='reverse')

    trainloader = DataLoader(dataset, batch_size=1)
    res=[]
    for i, (data, label) in enumerate(trainloader):
        # print(label)
        res.append(label)
        # img = torchvision.utils.make_grid(data).numpy()
        # # print img.shape
        # # print label.shape
        # # chw -> hwc
        # img = np.transpose(img, (1, 2, 0))
        # #cv2.imshow('img', img)
        # img *= np.array([0.5, 0.5, 0.5])*255
        # img += np.array([0.5, 0.5, 0.5])*255
        # #img += np.array([1, 1, 1])
        # #img *= 127.5
        # img = img.astype(np.uint8)
        # img = img[:, :, [2, 1, 0]]
        #
        # # cv2.imshow('img', img)
        # # cv2.waitKey(0)
        # # break
        # # dst.decode_segmap(labels.numpy()[0], plot=True)