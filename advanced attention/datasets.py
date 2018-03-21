import torchvision.transforms as transforms
import torch.utils.data as data
import PIL
import glob
import os
from PIL import Image

class CustomClusteringDataset(data.Dataset):

    def proper_transform(self):

        normal_transform = transforms.Compose([
            transforms.Scale([self.sizes[0],self.sizes[1]]),
            transforms.ToTensor(),
        ])

        return normal_transform
    def proper_untransform_playtype(self):

        # viewpoint means
        means = [0,0,0]
        stds = [1,1,1]
        normal_transform = transforms.Compose([
            transforms.Scale([self.sizes[0],self.sizes[1]]),
            transforms.ToTensor(),
            transforms.Normalize((means[0],means[1],means[2]),(stds[0],stds[1],stds[2]))
        ])
    def proper_untransform(self):

        # viewpoint means
        means = [-0.01 , -0.01 , -0.01]
        stds = [0.1 ,  0.1 , 0.1]
        normal_transform = transforms.Compose([
            transforms.Scale([self.sizes[0],self.sizes[1]]),
            transforms.ToTensor(),
            transforms.Normalize((means[0],means[1],means[2]),(stds[0],stds[1],stds[2]))
        ])

        return normal_transform

    def trainable(self, index):
        if index % self.datasplit[1] < self.datasplit[0]:
            if self.testing:
                return False
            else:
                return True
        if self.testing:
            return True
        else:
            return False
    
    def testable(self, index):
        return not self.trainable(index) 
    
    def __init__(self, path = None, sizes=(64,64), subset=0.1, datasplit=(10,10), testing=False):
        self.sizes = sizes
        self.path = path
        self.datasplit = datasplit
        self.testing = testing

        self.tun = self.proper_untransform()
        self.t = self.proper_transform()

        orig_maps = [s for s in glob.glob(self.path+"/sample*.*") if "orig" in s]
        saliency_maps = [s for s in glob.glob(self.path+"/sample*.*") if "orig" not in s]

        self.img_dict = {}

        j = 0
        for i, map in enumerate(saliency_maps):
            if i > len(saliency_maps)*subset:
                break
            basename = os.path.basename(map)
            mytype, mybatch, myindex, mylabel = basename.split(".")[0].split("_")
            if self.trainable(i):
                self.img_dict[j] = [None,None,int(mylabel), int(myindex)]
                self.img_dict[j][1]=map
                j += 1

        j = 0
        for i, map in enumerate(orig_maps):
            if i > len(saliency_maps)*subset:
                break
            basename = os.path.basename(map)
            _, mytype, mybatch, myindex, mylabel = basename.split(".")[0].split("_")
            if self.trainable(i):
                self.img_dict[j][0]=map
                j += 1
            
    
    def __getitem__(self, index):

        original_image_sequence = self.img_dict[index][0]
        saliency_image_sequence = self.img_dict[index][1]
        label_sequence = self.img_dict[index][2]

        img = self.tun(PIL.Image.open(original_image_sequence))
        img_s = self.t(PIL.Image.open(saliency_image_sequence).convert("L"))

        return img, img_s, label_sequence

    def __len__(self):
        return len(self.img_dict)


class CustomWord2Vec(data.Dataset):

    def proper_transform(self):

        normal_transform = transforms.Compose([
            transforms.Scale([self.sizes[0],self.sizes[1]]),
            transforms.ToTensor(),
        ])

        return normal_transform

    def proper_untransform(self):

        # viewpoint means
        means = [-0.01 , -0.01 , -0.01]
        stds = [0.1 ,  0.1 , 0.1]
        normal_transform = transforms.Compose([
            transforms.Scale([self.sizes[0],self.sizes[1]]),
            transforms.ToTensor(),
            transforms.Normalize((means[0],means[1],means[2]),(stds[0],stds[1],stds[2]))
        ])

        return normal_transform

    def __init__(self, path = "C:/Users/James/Desktop/Datasets/experiments/viewpoint_EB_final_data_proper/", sizes=(64,64), subset=1.0):
        self.sizes = sizes
        self.path = path

        self.tun = self.proper_untransform()
        self.tun_playtype = self.proper_untransform_playtype()
        self.t = self.proper_transform()

        orig_maps = [s for s in glob.glob(self.path+"/sample*.png") if "orig" in s]
        saliency_maps = [s for s in glob.glob(self.path+"/sample*.png") if "orig" not in s]

        self.img_dict = {}

        for i, map in enumerate(saliency_maps):
            if i > len(saliency_maps)*subset:
                break
            basename = os.path.basename(map)
            mytype, mybatch, myindex, mylabel = basename.strip(".png").split("_")
            self.img_dict[i] = [None,None,int(mylabel)]
            self.img_dict[i][1]=map

        for i, map in enumerate(orig_maps):
            if i > len(saliency_maps)*subset:
                break
            basename = os.path.basename(map)
            _, mytype, mybatch, myindex, mylabel = basename.strip(".png").split("_")
            self.img_dict[i][0]=map

    def __getitem__(self, index):

        original_image_sequence = self.img_dict[index][0]
        saliency_image_sequence = self.img_dict[index][1]
        label_sequence = self.img_dict[index][2]

        if 'playtype' in self.img_dict[index][0]:
            img = self.tun_playtype(PIL.Image.open(original_image_sequence))
        else:  
            img = self.tun(PIL.Image.open(original_image_sequence))
        img_s = self.t(PIL.Image.open(saliency_image_sequence).convert("L"))

        return img, img_s, label_sequence

    def __len__(self):
        return len(self.img_dict)