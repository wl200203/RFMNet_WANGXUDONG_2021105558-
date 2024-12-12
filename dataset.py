# coding=utf-8

import os
import cv2
import torch
import numpy as np
from Net import Net
import dataset as dataset
from torch.utils.data import Dataset

########################### Data Augmentation ###########################
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean 
        self.std = std
    
    def __call__(self, image, mask, edge=None):
        # import pdb; pdb.set_trace()
        
        image = (image - self.mean)/self.std
        mask /= 255
        if edge is None:
            return image, mask
        else:
            edge /= 255
            return image, mask, edge


class RandomCrop(object):
    def __call__(self, image, mask, edge):
        H, W, _ = image.shape
        randw = np.random.randint(W/8)
        randh = np.random.randint(H/8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H+offseth-randh, offsetw, W+offsetw-randw
        return image[p0:p1, p2:p3, :], mask[p0:p1, p2:p3], edge[p0:p1, p2:p3]


class RandomFlip(object):
    def __call__(self, image, mask, edge):
        if np.random.randint(2) == 0:
            return image[:, ::-1, :], mask[:, ::-1], edge[:, ::-1]
        else:
            return image, mask, edge


class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return image, mask

class ToTensor(object):
    def __call__(self, image, mask):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        mask = torch.from_numpy(mask)
        return image, mask


########################### Config File ###########################
class Config(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
            
        if not hasattr(self, 'snapshot'):
            self.snapshot = None  # 默认值可以是 None，或其他适合的默认路径
        #self.kwargs = kwargs
        # import pdb;pdb.set_trace()
        
        if self.dataset == 'MSD':
            # MSD
            self.mean = np.array([[[136.67972, 128.13225, 119.58932]]])
            self.std = np.array([[[64.26382, 67.363945, 68.53635]]])
        elif self.dataset == 'PMD':
            # PMD
            self.mean = np.array([[[129.81274, 115.53708, 100.38846]]])
            self.std = np.array([[[65.67121, 65.61214, 67.4352]]])
        else:
            raise ValueError("Unknown dataset specified.")
            #return None
        # import pdb;pdb.set_trace()
        print('\nParameters...')
        for k, v in kwargs.items():
            print('%-10s: %s' % (k, v))

    #def __getattr__(self, name):
     #   if name in self.kwargs:
      #      return self.kwargs[name]
       # else:
        #    return None


########################### Dataset Class ###########################
                
class Data(Dataset):
    #import pdb;pdb.set_trace()
    def __init__(self, cfg):
        self.cfg = cfg
        self.normalize = Normalize(mean=cfg.mean, std=cfg.std)
        self.randomcrop = RandomCrop()
        self.randomflip = RandomFlip()
        self.resize = Resize(352, 352)
        self.totensor = ToTensor()

        # 自动读取图片文件夹中的所有文件
        # image_dir = kwargs.predict_path
        image_dir = os.path.join(cfg.datapath, cfg.mode, 'images')
        #os.path.join(cfg.datapath, cfg.mode, 'images')
        self.samples = [os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
        # import pdb;pdb.set_trace

    def __getitem__(self, idx):
        name = self.samples[idx]

        # 读取图像和 mask
        image = cv2.imread(self.cfg.datapath+'/'+self.cfg.mode+'/images/'+name+'.jpg')[:,:,::-1].astype(np.float32)
        mask = cv2.imread(self.cfg.datapath+'/'+self.cfg.mode+'/masks/'+name+'.png', 0).astype(np.float32)
        shape = mask.shape
        # import pdb;pdb.set_trace()
        if self.cfg.mode == 'train':
            # 生成边缘图像
            gt = mask > 0
            gt = gt.astype(np.float64)
            gy, gx = np.gradient(gt)
            temp_edge = gy**2 + gx**2
            temp_edge[temp_edge != 0] = 1
            bound = (temp_edge * 255).astype(np.uint8)
            
            # 保存边缘图像到指定文件夹
            # edge_output_dir = os.path.join(self.cfg.datapath, self.cfg.mode, 'edge')
            # os.makedirs(edge_output_dir, exist_ok=True)
            # edge_path = os.path.join('/home/gpuadmin/hds/mirror_rorate/HetNet/dataset/msd_edge/', name + '.png')
            # cv2.imwrite(edge_path, bound)

            # 读取刚刚生成的边缘图像
            edge = bound.astype(np.float32)
            # import pdb;pdb.set_trace()
            # 数据增广
            
            image, mask, edge = self.normalize(image, mask, edge)
            image, mask, edge = self.randomcrop(image, mask, edge)
            image, mask, edge = self.randomflip(image, mask, edge)

            return image, mask, edge
        else:
            # import pdb;pdb.set_trace()
            image, mask = self.normalize(image, mask)
            image, mask = self.resize(image, mask)
            image, mask = self.totensor(image, mask)
            
            return image, mask, shape, name
        
        

    def collate(self, batch):
        if self.cfg.dataset == 'MSD':
            size = [224, 256, 288, 320, 352][np.random.randint(0, 5)] # MSD
        elif self.cfg.dataset == 'PMD':
            size = [288, 320, 352, 384, 416][np.random.randint(0, 5)] # PMD
        else:
            return None  
            
        image, mask, edge = [list(item) for item in zip(*batch)]
        for i in range(len(batch)):
            image[i] = cv2.resize(image[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            mask[i] = cv2.resize(mask[i],  dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            edge[i] = cv2.resize(edge[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
        image = torch.from_numpy(np.stack(image, axis=0)).permute(0, 3, 1, 2)
        mask = torch.from_numpy(np.stack(mask, axis=0)).unsqueeze(1)
        edge = torch.from_numpy(np.stack(edge, axis=0)).unsqueeze(1)
        return image, mask, edge

    def __len__(self):
        return len(self.samples)
