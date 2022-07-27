import glob
import json
import os
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset,DataLoader

class mydata(Dataset):

    def __init__(self, root ,train=True, transform = None, target_transform=None):
        super(mydata, self).__init__()
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        #如果是训练则加载训练集，如果是测试则加载测试集
        if self.train :
            self.path = root+'/train/'
        else:
            self.path = root+'/test/'
            #self.path = root + '/test/'
        self.img_list = os.listdir(self.path)
    def __getitem__(self, index):
        str0 = "("
        self.img_path = self.path + self.img_list[index]
        image = cv2.imread(self.img_path)
        # image=cv2.resize(image,(224,224))
        img = Image.fromarray(image)
        num = self.img_list[index].index(str0)
        label_name = self.img_list[index][:num].strip()
        #print(label_name)
        if label_name == "Orah mandarin":
            label = 0
        elif label_name == "Shiranui tangor":
            label = 1
        elif label_name == "Murcott tangerine":
            label = 2
        elif label_name == "Jinyou mandarin":
            label = 3
        elif label_name == "Haruka mandarin":
            label = 4
        elif label_name == "Caracara navel orange":
            label = 5
        elif label_name == "Tarocco blood orange":
            label = 6
        elif label_name == "Jincheng sweet orange":
            label = 7
        else:
            print("data label error")
        label = int(label)
        if self.transform is not None:
            img=self.transform(img)
        if self.target_transform is not None:
            label=self.target_transform(label)
        return img, label

    def __len__(self):
        return len(self.img_list)
