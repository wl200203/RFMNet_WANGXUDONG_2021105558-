# coding=utf-8

import os
import time
import sys
sys.path.insert(0, '../')
sys.dont_write_bytecode = True
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import dataset as dataset
import argparse



def get_network(net_name):
    # import pdb; pdb.set_trace()
    if net_name == "Net":
        from Net import Net
    elif net_name == "Net_gan":
        from Net_gan import Net
    elif net_name == "Net_xudong":
        from Net_xudong import Net
    elif net_name == "Net_gan_GNN":
        from Net_gan_GNN import Net    
    else:
        raise ValueError(f"Unknown network name: {net_name}")
    return Net


class Test(object):
    def __init__(self, Dataset, Network, path,snapshot ,save_path='./Test_model_result'):
        ## dataset
        self.cfg = Dataset.Config(dataset='MSD', datapath=path, snapshot=snapshot, mode='test')
        self.data = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=8)
        ## network
        self.net = Network(self.cfg)
        self.net.train(False)
        self.net.cuda()
        self.save_path = save_path
        # import pdb;pdb.set_trace()
    def save(self):
        current_date = time.strftime("%Y-%m-%d_%H-%M")  
        model_name = self.cfg.snapshot.split('/')[-1]  # 提取模型名
        # import pdb;pdb.set_trace()
        save_path = f'./Test_model_result/{current_date}_{model_name}/MSD/' + self.cfg.datapath.split('/')[-1]
        save_edge = f'./Test_model_result/{current_date}_{model_name}/Edge/' + self.cfg.datapath.split('/')[-1]


        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(save_edge):
            os.makedirs(save_edge)

        with torch.no_grad():
            cost_time = list()
            # han = []
            for image, mask, shape, name in self.loader:

                image = image.cuda().float()
                torch.cuda.synchronize()
                start_time = time.perf_counter()
                out, out_edge = self.net(image, shape)
                # out, out_edge = self.net(image)
                torch.cuda.synchronize()
                # import pdb;pdb.set_trace()
                cost_time.append(time.perf_counter() - start_time)
                pred = (torch.sigmoid(out[0, 0]) * 255).cpu().numpy()
                edge = (torch.sigmoid(out_edge[0, 0]) * 255).cpu().numpy()

                # 动态生成保存路径

                # import pdb;pdb.set_trace()
                # han.append(name[0])
                # continue
                cv2.imwrite(save_path+'/'+name[0]+'.png', np.round(pred))
                cv2.imwrite(save_edge + '/' + name[0] + '_edge.png', np.round(edge))
            # import pdb;pdb.set_trace()
            cost_time.pop(0)
            print('Mean running time is: ', np.mean(cost_time))
            print("FPS is: ", len(self.loader.dataset) / np.sum(cost_time))
            
            
            
            
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='MSD')
    parser.add_argument('--datapath', default='/home/gpuadmin/hds/SAM-Adapter-PyTorch/load/MSD_source')
    parser.add_argument('--snapshot', default='/home/gpuadmin/hds/mirror_rorate/HetNet/outputs/metrics_models/model_best_mae_primitive')
    parser.add_argument('--save_path', default='./MSD-test-output')
    parser.add_argument('--net_name', type=str, required=True, help='Name of the network to use (e.g., Net, Net_gan, Net_xudong,Net_gan_GNN)')
    args = parser.parse_args()

    Net = get_network(args.net_name)

    for path in [args.datapath]:
        test = Test(dataset, Net, path, save_path=args.save_path, snapshot=args.snapshot)
        test.save()

        
        
    