# coding=utf-8
import sys
import datetime
import random
import time
import logging    #10.31
import numpy as np  
sys.path.insert(0, '../')
sys.dont_write_bytecode = True
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import dataset as dataset
import argparse
import os
import shutil
from torchvision.utils import save_image

def save_specified_py_file(file_path, code_dir):
    """
    将指定的 .py 文件复制保存到目标文件夹。
    
    参数:
        file_path: 需要保存的 .py 文件的路径
        code_dir: 保存的目标文件夹
    """
    if not os.path.exists(code_dir):
        os.makedirs(code_dir)

    try:
        if os.path.exists(file_path) and file_path.endswith('.py'):
            # 获取文件名
            file_name = os.path.basename(file_path)
            # 目标路径
            dest_path = os.path.join(code_dir, file_name)
            # 复制文件
            shutil.copy(file_path, dest_path)
            print(f"文件 {file_name} 已保存到: {dest_path}")
        else:
            print(f"文件 {file_path} 不存在或不是一个 .py 文件")
    except Exception as e:
        print(f"保存文件 {file_path} 时出错: {e}")

# 自动创建运行文件夹结构
def create_output_folders(sh_filename=None):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  # 带时间戳的文件夹名
    
    if sh_filename:
        sh_name = os.path.basename(sh_filename).split(".")[0]  # 提取.sh文件的名称（不含扩展名）
        base_output_dir = f'./output_{timestamp}_{sh_name}'  # 根目录名加入.sh文件名
    else:
        base_output_dir = f'./output_{timestamp}'  # 根目录名只用时间戳
    os.makedirs(base_output_dir, exist_ok=True)

    sub_dirs = {
        'models': os.path.join(base_output_dir, 'models'),
        'log': os.path.join(base_output_dir, 'log'),
        'code': os.path.join(base_output_dir, 'code'),
        'loss': os.path.join(base_output_dir, 'loss'),
    }

    for key, sub_dir in sub_dirs.items():
        os.makedirs(sub_dir, exist_ok=True)

    # 打印创建的文件夹结构
    print(f"创建了以下文件夹结构：")
    print(f"根目录: {base_output_dir}")
    for key, sub_dir in sub_dirs.items():
        print(f"{key}: {sub_dir}")

    return base_output_dir, sub_dirs

# choose Net
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



def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def bce_loss(pred, mask):
    pred = torch.sigmoid(pred)
    bce_loss = nn.BCELoss()
    bce = bce_loss(pred, mask)
    #import pdb; pdb.set_trace()
    return bce


def validate(model, val_loader, nums):
    model.train(False)
    avg_mae = 0.0
    with torch.no_grad():
        for image, mask, shape, name in val_loader:
            image, mask = image.cuda().float(), mask.cuda().float()
            out, _, _, _, _, _ = model(image)
            pred = torch.sigmoid(out[0, 0])
            #avg_mae += torch.abs(pred - mask[0]).mean()
            avg_mae += torch.mean(abs(pred - mask[0])).item()
            #pred = torch.sigmoid(out)
            #avg_mae += compute_mae(pred, mask[0])
    model.train(True)
    return (avg_mae / nums)
    

def validate_save_image(model, val_loader, nums):
    model.train(False)
    avg_mae = 0.0
    # 定义保存目录
    save_dir = "./output_images/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with torch.no_grad():
        for image, mask, shape, name in val_loader:
            image, mask = image.cuda().float(), mask.cuda().float()
            out, _, _, _, _, _ = model(image)
            pred = torch.sigmoid(out[0, 0])
            # import pdb; pdb.set_trace()
            #avg_mae += torch.abs(pred - mask[0]).mean()
            avg_mae += torch.mean(abs(pred - mask[0])).item()
            #pred = torch.sigmoid(out)
            #avg_mae += compute_mae(pred, mask[0])
            pred_image_path = os.path.join(save_dir, f"{name[0]}.png")
            save_image(pred, pred_image_path)
    model.train(True)
    return (avg_mae / nums)


# 计算 mIoU 的验证函数
def validate_miou(model, val_loader, nums):
    model.train(False)  # 切换到评估模式，关闭 Dropout 等层
    total_inter = 0.0
    total_union = 0.0
    with torch.no_grad():
        for image, mask, shape, name in val_loader:
            image, mask = image.cuda().float(), mask.cuda().float()
            out, _, _, _, _, _ = model(image)
            pred = torch.sigmoid(out[0, 0])  # 使用 sigmoid 将输出限制在 0 到 1 之间
            pred = (pred >= 0.5).float()  # 将预测值二值化，阈值为 0.5

            # 计算交集和并集
            intersection = (pred * mask[0]).sum()
            union = pred.sum() + mask[0].sum() - intersection

            total_inter += intersection.item()
            total_union += union.item()
    # import pdb;pdb.set_trace()
    model.train(True)  # 恢复到训练模式
    # 返回平均的 IoU
    miou= total_inter / (total_union + 1e-6)  # 避免除以零
    return miou




# 训练函数
def train(Dataset, Network, cfg,args,sub_dirs):
    # 设置随机种子
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    seed = 7
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    device=('cuda')


    # 数据集
    data = Dataset.Data(cfg)
    loader = DataLoader(data, collate_fn=data.collate, batch_size=args.batch_size, shuffle=True, num_workers=8)
    #loader = DataLoader(data, collate_fn=data.collate, batch_size=16, shuffle=True, num_workers=4)#
    # 验证集加载
    val_cfg = Dataset.Config(dataset=cfg.dataset,datapath=cfg.datapath, mode='test')
    val_data = Dataset.Data(val_cfg)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0)
    #val_loader = DataLoader(val_data, batch_size=, shuffle=False, num_workers=4)# 

    # 网络模型
    net = Network(cfg)
    # net.cuda()
    net.to(device)

    # net = torch.nn.DataParallel(net1)
    # net = net.cuda()            # 将模型迁移到 GPU
    
    
    net.train(True)

    # 优化器
    enc_params, dec_params = [], []
    for name, param in net.named_parameters():
        if 'bkbone' in name:
            enc_params.append(param)
        else:
            dec_params.append(param)
            
            
            
    optimizer = torch.optim.SGD([{'params': enc_params}, {'params': dec_params}], lr=cfg.lr, momentum=cfg.momen, weight_decay=cfg.decay, nesterov=True)




    sw = SummaryWriter(log_dir=sub_dirs['loss'])
    global_step = 1
    # best_miou = 0.0        #用于记录最简mIOU
    min_mae = float('inf')  # 用于记录最小 MAE

    # 开始训练
    for epoch in range(cfg.epoch):
        optimizer.param_groups[0]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr*0.1
        optimizer.param_groups[1]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr
        # count =0
        # import pdb;pdb.set_trace()
        for step, (image, mask, edge) in enumerate(loader):
            # import pdb;pdb.set_trace()
            # break
            image, mask, edge = image.to(device).float(), mask.to(device).float(), edge.float().to(device)
            # count +=1          
            # continue
            image = image.to(device)
            out1, out_edge1, out2,  out3, out4, out5 = net(image)
            loss1 = structure_loss(out1, mask)
            #import pdb; pdb.set_trace()
            loss_edge = bce_loss(out_edge1, edge)
            #import pdb; pdb.set_trace()
            loss2 = structure_loss(out2, mask)
            loss3 = structure_loss(out3, mask)
            loss4 = structure_loss(out4, mask)
            loss5 = structure_loss(out5, mask)
            loss = loss1 + loss_edge + loss2/2 + loss3/4 + loss4/8 + loss5/16
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
            global_step += 1
            sw.add_scalar('lr', optimizer.param_groups[1]['lr'], global_step=global_step)
            sw.add_scalars('loss', {'loss1': loss1.item(), 'loss_edge1': loss_edge.item(), 'loss2': loss2.item(),
                                    'loss3': loss3.item(), 'loss4': loss4.item(), 'loss5': loss5.item()}, global_step=global_step)

            if step % 10 == 0:
                print('%s | step:%d/%d/%d | lr=%.6f | loss=%.6f' % (datetime.datetime.now(), global_step, epoch+1, cfg.epoch, optimizer.param_groups[1]['lr'], loss.item()))

        if epoch+1 > cfg.epoch//2 or (epoch+1) % 2 == 0:
            #miou = validate_miou(net, val_loader, 955) #IOU计算
            mae = validate(net, val_loader, 955)  # 新增 MAE 计算
            # logging.info('Epoch %d: MSD mIoU = %.6f, MAE = %.6f', epoch + 1, mae)
            # logging.info('Epoch %d: MAE = %.6f', epoch + 1, mae)

            # if miou > best_miou:
            #     best_miou = miou
            #     model_path = os.path.join(sub_dirs['models'], 'best_miou.pth')
            #     torch.save(net.state_dict(), model_path)
            #     print(f"Epoch {epoch}, 新最佳mIoU: {miou:.6f}，保存至 {model_path}")

            # with open(os.path.join(sub_dirs['log'], 'training_log.txt'), 'a') as log_file:
            #     log_file.write(f"Epoch {epoch}, mIoU: {miou}, 最佳mIoU: {best_miou}\n")
                
            if mae < min_mae:
                min_mae = mae
                model_path_mae = os.path.join(sub_dirs['models'], 'best_mae.pth')
                torch.save(net.state_dict(), model_path_mae)
                print(f"Epoch {epoch}, 新最小 MAE: {mae:.6f}，保存至 {model_path_mae}")
            # import pdb;pdb.set_trace()
            # 写入日志文件
            with open(os.path.join(sub_dirs['log'], 'training_log.txt'), 'a') as log_file:
                log_file.write(f"Epoch {epoch}, MAE: {mae:.6f}, 最小 MAE: {min_mae:.6f}\n")

        
            with open(os.path.join(sub_dirs['loss'], f'loss_epoch_{epoch + 1}.txt'), 'w') as loss_file:
                loss_file.write(f"Epoch {epoch}, Loss: {loss.item()}\n")

                
                

if __name__ == '__main__':
    # EXP_NAME = 'check-msd'
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='MSD')
    parser.add_argument('--mode', default='test')
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momen', type=float, default=0.9)
    parser.add_argument('--decay', type=float, default=5e-4)
    parser.add_argument('--epoch', type=int, default=150)
    parser.add_argument('--backbone_path', default='./resnext/resnext_101_32x4d.pth')
    parser.add_argument('--net_name', type=str, required=True, help='Name of the network to use (e.g., Net, Net_gan, Net_xudong)')
    parser.add_argument('--datapath', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--savepath', default='./check-msd/')
    parser.add_argument('--sh_filename', type=str, required=True, help='Name of the .sh file used to start the script')  
    args = parser.parse_args() 
    base_output_dir, sub_dirs = create_output_folders(args.sh_filename)
    
    log_file_path = os.path.join(sub_dirs['log'], 'training_log.txt')
    logging.basicConfig(
        filename=log_file_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    file_to_save = '/home/gpuadmin/hds/mirror_rorate/HetNet/Rotated/adaptive_rotated_conv.py'
    # 验证文件路径
    if not os.path.exists(file_to_save):
        print(f"文件路径错误，文件不存在: {file_to_save}")
    else:
        print(f"文件路径正确: {file_to_save}")

    # 验证目标目录
    if not os.path.exists(sub_dirs['code']):
        print(f"目标目录不存在: {sub_dirs['code']}，正在创建...")
        os.makedirs(sub_dirs['code'], exist_ok=True)
    else:
        print(f"目标目录存在: {sub_dirs['code']}")

    # 保存文件
    print("开始保存指定文件...")
    save_specified_py_file(file_to_save, sub_dirs['code'])
    print("文件保存完成")
        
    
    # 保存运行时的 .sh 文件
    sh_file_path = os.path.join(sub_dirs['log'], 'run_command.sh')
    with open(sh_file_path, 'w') as sh_file:
        sh_file.write("#!/bin/bash\n\n")
        sh_file.write(f"python {__file__} \\\n")
        for arg, value in vars(args).items():
            sh_file.write(f"    --{arg} {value} \\\n")
        sh_file.write("\n")
    os.chmod(sh_file_path, 0o755) 
    print(f"运行命令已保存到: {sh_file_path}")

        
    Net = get_network(args.net_name)
    
    # 动态复制调用的网络文件到 `code` 文件夹
    net_file_map = {
        "Net": "Net.py",
        "Net_gan": "Net_gan.py",
        "Net_xudong": "Net_xudong.py",
        "Net_gan_GNN": "Net_gan_GNN.py"
    }
    # additional_file = "adaptive_rotated_conv.py" 
    
    
    if args.net_name in net_file_map:
        net_file_path = os.path.join(os.getcwd(), net_file_map[args.net_name])
        shutil.copy(net_file_path, sub_dirs['code'])
        print(f"已复制网络文件: {net_file_map[args.net_name]} 到 {sub_dirs['code']}")
    else:
        raise ValueError(f"Unknown network name: {args.net_name}")
    
    cfg = dataset.Config(
        dataset=args.dataset,
        datapath=args.datapath,
        net_name=args.net_name,
        mode=args.mode,
        batch=args.batch_size,
        lr=args.lr,
        momen=args.momen,
        decay=args.decay,
        epoch=args.epoch,
        savepath=args.savepath,
        backbone_path=args.backbone_path, 
    )
    
    # 复制当前代码到 `code` 文件夹
    shutil.copy(__file__, sub_dirs['code'])
    train(dataset, Net, cfg,args, sub_dirs)

