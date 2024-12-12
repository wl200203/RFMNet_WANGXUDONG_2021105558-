import torch
import torch.nn as nn
import torch.nn.functional as F
from resnext.resnext101_regular import ResNeXt101 
from Rotated import *               #导如rotate模块
from Rotated.adaptive_rotated_conv import AdaptiveRotatedConv2d_flip,AdaptiveRotatedConv2d_fusion
from Rotated import RountingFunction_flip


#from HetNet.Rotated.adaptive_rotated_conv import AdaptiveRotatedConv2d, AdaptiveRotatedConv2d_flip
from GNN_gan import GNN, SpatialGNN           #GNN库导入
#from Rotated.adaptive_rotated_conv import GatingModule


def _get_rotation_matrix(thetas):
    #import pdb; pdb.set_trace()#这段代码根据输入角度的正负性，选择正旋转矩阵或负旋转矩阵，生成一个适用于整个批次和组的旋转矩阵张量。
    bs, g = thetas.shape
    device = thetas.device
    thetas = thetas.reshape(-1)  # [bs, n] --> [bs x n]    将张量展平成一维，目的是为了方便后续对每个角度进行处理（例如计算旋转矩阵）
    
    x = torch.cos(thetas)
    y = torch.sin(thetas)
    x = x.unsqueeze(0).unsqueeze(0)  # shape = [1, 1, bs * g]
    y = y.unsqueeze(0).unsqueeze(0)
    a = x - y
    b = x * y
    c = x + y

    rot_mat_positive = torch.cat((  #dim=1 表示沿着列的方向拼接元素，构造矩阵的一行。
        torch.cat((a, 1-a, torch.zeros(1, 7, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 1, bs*g, device=device), x-b, b, torch.zeros(1, 1, bs*g, device=device), 1-c+b, y-b, torch.zeros(1, 3, bs*g, device=device)), dim=1), 
        torch.cat((torch.zeros(1, 2, bs*g, device=device), a, torch.zeros(1, 2, bs*g, device=device), 1-a, torch.zeros(1, 3, bs*g, device=device)), dim=1),
        torch.cat((b, y-b, torch.zeros(1,1 , bs*g, device=device), x-b, 1-c+b, torch.zeros(1, 4, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 4, bs*g, device=device), torch.ones(1, 1, bs*g, device=device), torch.zeros(1, 4, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 4, bs*g, device=device), 1-c+b, x-b, torch.zeros(1, 1, bs*g, device=device), y-b, b), dim=1),
        torch.cat((torch.zeros(1, 3, bs*g, device=device), 1-a, torch.zeros(1, 2, bs*g, device=device), a, torch.zeros(1, 2, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 3, bs*g, device=device), y-b, 1-c+b, torch.zeros(1, 1, bs*g, device=device), b, x-b, torch.zeros(1, 1, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 7, bs*g, device=device), 1-a, a), dim=1)
    ), dim=0)  # shape = [k^2, k^2, bs*g]

    rot_mat_negative = torch.cat((
        torch.cat((c, torch.zeros(1, 2, bs*g, device=device), 1-c, torch.zeros(1, 5, bs*g, device=device)), dim=1),
        torch.cat((-b, x+b, torch.zeros(1, 1, bs*g, device=device), b-y, 1-a-b, torch.zeros(1, 4, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 1, bs*g, device=device), 1-c, c, torch.zeros(1, 6, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 3, bs*g, device=device), x+b, 1-a-b, torch.zeros(1, 1, bs*g, device=device), -b, b-y, torch.zeros(1, 1, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 4, bs*g, device=device), torch.ones(1, 1, bs*g, device=device), torch.zeros(1, 4, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 1, bs*g, device=device), b-y, -b, torch.zeros(1, 1, bs*g, device=device), 1-a-b, x+b, torch.zeros(1, 3, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 6, bs*g, device=device), c, 1-c, torch.zeros(1, 1, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 4, bs*g, device=device), 1-a-b, b-y, torch.zeros(1, 1, bs*g, device=device), x+b, -b), dim=1),
        torch.cat((torch.zeros(1, 5, bs*g, device=device), 1-c, torch.zeros(1, 2, bs*g, device=device), c), dim=1)
    ), dim=0)  # shape = [k^2, k^2, bs*g]

    mask = (thetas >= 0).unsqueeze(0).unsqueeze(0)    #thetas 是输入的角度张量，形状为 [bs * g],通过调用 unsqueeze(0) 两次，将 mask 的形状从 [bs * g] 扩展到 [1, 1, bs * g]。这样做的目的是方便后续使用广播机制与旋转矩阵相乘。
    mask = mask.float()                                                   # shape = [1, 1, bs*g]
    rot_mat = mask * rot_mat_positive + (1 - mask) * rot_mat_negative     # shape = [k*k, k*k, bs*g]
    rot_mat = rot_mat.permute(2, 0, 1)                                    # shape = [bs*g, k*k, k*k]
    rot_mat = rot_mat.reshape(bs, g, rot_mat.shape[1], rot_mat.shape[2])  # shape = [bs, g, k*k, k*k]
    return rot_mat   #最终返回的 rot_mat 是形状为 [bs, g, k*k, k*k] 的旋转矩阵张量，表示每个批次的每个组对应的卷积核旋转矩阵


def batch_rotate_multiweight(weights, lambdas, thetas):  #表示经过旋转后的卷积核权重
    #weights：形状：[kernel_number, Cout, Cin, k, k] kernel_number卷积核数量
    """
    Let
       = = b
        kernel_number = n
        kernel_size = 3
    Args:
        weights: tensor, shape = [kernel_number, Cout, Cin, k, k]
        thetas: tensor of thetas,  shape = [batch_size, kernel_number]
    Return:
        weights_out: tensor, shape = [batch_size x Cout, Cin // groups, k, k]
    """
    assert(thetas.shape == lambdas.shape)
    assert(lambdas.shape[1] == weights.shape[0])

    b = thetas.shape[0]
    n = thetas.shape[1]
    k = weights.shape[-1]
    _, Cout, Cin, _, _ = weights.shape

    if k == 3 :
        # Stage 1:
        # input: thetas: [b, n]
        #        lambdas: [b, n]
        # output: rotation_matrix: [b, n, 9, 9] (with gate) --> [b*9, n*9]

        #       Sub_Stage 1.1:
        #       input: [b, n] kernel
        #       output: [b, n, 9, 9] rotation matrix
        rotation_matrix = _get_rotation_matrix(thetas)

        #       Sub_Stage 1.2:
        #       input: [b, n, 9, 9] rotation matrix
        #              [b, n] lambdas
        #          --> [b, n, 1, 1] lambdas
        #          --> [b, n, 1, 1] lambdas dot [b, n, 9, 9] rotation matrix
        #          --> [b, n, 9, 9] rotation matrix with gate (done)
        #       output: [b, n, 9, 9] rotation matrix with gate
        lambdas = lambdas.unsqueeze(2).unsqueeze(3)
        rotation_matrix = torch.mul(rotation_matrix, lambdas)

        #       Sub_Stage 1.3: Reshape
        #       input: [b, n, 9, 9] rotation matrix with gate
        #       output: [b*9, n*9] rotation matrix with gate
        rotation_matrix = rotation_matrix.permute(0, 2, 1, 3)
        rotation_matrix = rotation_matrix.reshape(b*k*k, n*k*k)

        # Stage 2: Reshape
        # input: weights: [n, Cout, Cin, 3, 3]
        #             --> [n, 3, 3, Cout, Cin]
        #             --> [n*9, Cout*Cin] done
        # output: weights: [n*9, Cout*Cin]
        weights = weights.permute(0, 3, 4, 1, 2)
        weights = weights.contiguous().view(n*k*k, Cout*Cin)


        # Stage 3: torch.mm
        # [b*9, n*9] x [n*9, Cout*Cin]
        # --> [b*9, Cout*Cin]
        weights = torch.mm(rotation_matrix, weights)

        # Stage 4: Reshape Back
        # input: [b*9, Cout*Cin]
        #    --> [b, 3, 3, Cout, Cin]
        #    --> [b, Cout, Cin, 3, 3]
        #    --> [b * Cout, Cin, 3, 3] done
        # output: [b * Cout, Cin, 3, 3]
        weights = weights.contiguous().view(b, k, k, Cout, Cin)
        weights = weights.permute(0, 3, 4, 1, 2)
        weights = weights.reshape(b * Cout, Cin, k, k)
    else:
        thetas = thetas.reshape(-1)  # [bs, n] --> [bs x n]

        x = torch.cos(thetas)
        y = torch.sin(thetas)
        rotate_matrix = torch.tensor([[x, -y, 0], [y, x, 0]])
        rotate_matrix = rotate_matrix.unsqueeze(0).repeat(n, 1, 1)

        weights = weights.contiguous().view(n, Cout*Cin, k, k)

        grid = F.affine_grid(rotate_matrix, weights.shape)
        weights = F.grid_sample(weights, grid, mode='biliner')

    return weights

############################################ Initialization ##############################################
def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)) or isinstance(m, nn.GroupNorm):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU) or isinstance(m, nn.MaxPool2d) or isinstance(m, nn.Softmax) or isinstance(m, nn.Sigmoid) or isinstance(m, nn.AdaptiveAvgPool2d) or isinstance(m, nn.ReLU6):
            pass
        elif isinstance(m,nn. ModuleList):
            weight_init(m)
        else:
            m.initialize()

############################################ Basic ##############################################
class basicConv(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(basicConv, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)
    
    def initialize(self):
        weight_init(self)


class RotatedBasicConv(nn.Module): 

    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True,
                kernel_number=1, rounting_func=None, rotate_func=batch_rotate_multiweight):
        super(RotatedBasicConv, self).__init__()
        
        conv = [AdaptiveRotatedConv2d(
            in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias,
            kernel_number=kernel_number, rounting_func=rounting_func, rotate_func=rotate_func
        )]
        
        if bn:
            #conv.append(nn.BatchNorm2d(out_channel))
            conv.append(nn.BatchNorm2d(out_channel))
        #import pdb; pdb.set_trace()
        if relu:
            conv.append(nn.ReLU())
        
        self.conv = nn.Sequential(*conv)



    def forward(self, x):
        # import pdb; pdb.set_trace()
        return self.conv(x)
    
    def initialize(self):
        weight_init(self)
        
        
class RotatedBasicConv_flip(nn.Module): 

    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True,
                kernel_number=1, rounting_func=None, rotate_func=batch_rotate_multiweight):
        super(RotatedBasicConv_flip, self).__init__()
        
        conv = [AdaptiveRotatedConv2d_flip(
            in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias,
            kernel_number=kernel_number, rounting_func=rounting_func, rotate_func=rotate_func
        )]
        
        if bn:
            #conv.append(nn.BatchNorm2d(out_channel))
            conv.append(nn.BatchNorm2d(out_channel))
        #import pdb; pdb.set_trace()
        if relu:
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        
        # self.conv_flip = nn.Sequential([AdaptiveRotatedConv2d(
        #     in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias,``
        #     kernel_number=kernel_number, rounting_func=rounting_func, rotate_func=rotate_func
        # ),nn.BatchNorm2d(out_channel) ])
        
        
    def forward(self, x):
        # import pdb; pdb.set_trace()
        return self.conv(x)
    
    def initialize(self):
        weight_init(self)

def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, dilation=dilation, bias=bias)

# Feature Fusion Module
class FFM(nn.Module):
    def __init__(self, channel):
        super(FFM, self).__init__()
        self.conv_1 = conv3x3(channel, channel)
        self.bn_1 = nn.BatchNorm2d(channel)
        self.conv_2 = conv3x3(channel, channel)
        self.bn_2 = nn.BatchNorm2d(channel)

    def forward(self, x_1, x_2):
        out = x_1 * x_2
        #out = torch.cat((x_1, x_2), dim=1)
        out = F.relu(self.bn_1(self.conv_1(out)), inplace=True)
        out = F.relu(self.bn_2(self.conv_2(out)), inplace=True)
        return out

    def initialize(self):
        weight_init(self)

# Cross Aggregation Module
class CAM(nn.Module):
    def __init__(self, channel):
        super(CAM, self).__init__()
        self.down = nn.Sequential(
            conv3x3(channel, channel, stride=2),
            nn.BatchNorm2d(channel)
        )
        self.conv_1 = conv3x3(channel, channel)
        self.bn_1 = nn.BatchNorm2d(channel)
        self.conv_2 = conv3x3(channel, channel)
        self.bn_2 = nn.BatchNorm2d(channel)
        self.mul = FFM(channel)

    def forward(self, x_high, x_low):
        left_1 = x_low
        left_2 = F.relu(self.down(x_low), inplace=True)
        right_1 = F.interpolate(x_high, size=x_low.size()[2:], mode='bilinear', align_corners=True)
        right_2 = x_high
        left = F.relu(self.bn_1(self.conv_1(left_1 * right_1)), inplace=True)
        right = F.relu(self.bn_2(self.conv_2(left_2 * right_2)), inplace=True)
        right = F.interpolate(right, size=x_low.size()[2:], mode='bilinear', align_corners=True)
        out = self.mul(left, right)
        return out

    def initialize(self):
        weight_init(self)

####################################### reflection semantic logical module (RSL) ##########################################
# Revised from: PraNet: Parallel Reverse Attention Network for Polyp Segmentation, MICCAI20
# https://github.com/DengPingFan/PraNet
class RFB_modified(nn.Module):                          #RSL
    '''reflection semantic logical module (RSL)'''
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            basicConv(in_channel, out_channel, 1, relu=False),
        )
        self.branch1 = nn.Sequential(
            basicConv(in_channel, out_channel, 1),
            basicConv(out_channel, out_channel, k=7, p=3),
            basicConv(out_channel, out_channel, 3, p=7, d=7, relu=False)
        )
        self.branch2 = nn.Sequential(
            basicConv(in_channel, out_channel, 1),
            basicConv(out_channel, out_channel, k=7, p=3),
            basicConv(out_channel, out_channel, k=7, p=3),
            basicConv(out_channel, out_channel, 3, p=7, d=7, relu=False)
        )

        self.conv_cat = basicConv(3*out_channel, out_channel, 3, p=1, relu=False)
        self.conv_res = basicConv(in_channel, out_channel, 1, relu=False)
        
        # self.rotated_flip_conv = AdaptiveRotatedConv2d_flip(
        #     in_channels=out_channel,
        #     out_channels=out_channel,
        #     kernel_size=3,
        #     stride=1,
        #     padding=1,
        #     kernel_number=4,
        #     rounting_func=RountingFunction_flip(in_channels=out_channel, kernel_number=4)
        # )

    def forward(self, x):#torch.Size([12, 2048, 11, 11])
        #import pdb; pdb.set_trace()
        x0 = self.branch0(x)#torch.Size([12, 64, 13, 13])
        x1 = self.branch1(x)#torch.Size([12, 64, 13, 13])
        x2 = self.branch2(x)#torch.Size([12, 64, 13, 13])
        #x3 = self.branch3(x)#torch.Size([12, 64, 13, 13])
        x_cat = self.conv_cat(torch.cat((x0, x1, x2), 1))

        x = self.relu(x_cat + self.conv_res(x))#torch.Size([12, 2048, 11, 11])
        return x#torch.Size([12, 64, 13, 13])
        
    def initialize(self):
        weight_init(self)
    

########################################### multi-orientation intensity-based contrasted module #########################################
class h_sigmoid(nn.Module):                       #优化sigmoid激活函数
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

    def initialize(self):
        weight_init(self)

class h_swish(nn.Module):                                #h_swish 是 h_sigmoid 和输入的乘积
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

    def initialize(self):
        weight_init(self)

class CoordAtt(nn.Module):              #坐标注意力机制
    # Revised from: Coordinate Attention for Efficient Mobile Network Design, CVPR21
    # https://github.com/houqb/CoordAttention
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        identity = x
        
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out
    
    def initialize(self):
        weight_init(self)
        
class SAM(nn.Module):                 #SAM（Split Attention Module）通过分组注意力机制对输入特征进行增强。它将输入特征沿通道维度分成多个组，每组独立进行注意力处理，并最终将增强后的特征组拼接在一起
    def __init__(self, nin: int, nout: int, num_splits: int) -> None:
        super(SAM, self).__init__()

        assert nin % num_splits == 0

        self.nin = nin
        self.nout = nout
        self.num_splits = num_splits

        self.subspaces = nn.ModuleList(
            [CoordAtt(int(self.nin / self.num_splits),int(self.nin / self.num_splits)) for i in range(self.num_splits)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        group_size = int(self.nin / self.num_splits)

        # split at batch dimension
        sub_feat = torch.chunk(x, self.num_splits, dim=1)

        out = []
        for idx, l in enumerate(self.subspaces):
            out.append(self.subspaces[idx](sub_feat[idx]))

        out = torch.cat(out, dim=1)

        return out

    def initialize(self):
        weight_init(self)


class IntensityPositionModule(nn.Module):        #ICFE模块 rotate90°操作  调用SAM进行平均池化操作   修改
    '''multi-orientation intensity-based contrasted module (MIC)'''
    def __init__(self, inplanes, outplanes, g=1):
        super(IntensityPositionModule, self).__init__()

        # self.SA1 = SAM(inplanes, outplanes, g)
        # self.SA2 = SAM(inplanes, outplanes, g)
        routing_function1 =  RountingFunction(in_channels=inplanes, kernel_number=4)
        #import pdb; pdb.set_trace()
        self.conv = nn.Sequential(
                # RotatedBasicConv_k4(inplanes, inplanes, k=3, s=1, p=1, d=1, g=inplanes,rounting_func = routing_function1),
                RotatedBasicConv(inplanes, inplanes, k=3, s=1, p=1, d=1,rounting_func = routing_function1,kernel_number=4),
                #basicConv(inplanes, outplanes, k=1, s=1, p=0, relu = True)
                basicConv(inplanes, outplanes, k=1, s=1, p=0, relu = True)
                )
    def forward(self, x):
        fused_feature = self.conv(x)  # 如果 RotatedBasicConv 已处理 loss，这里无需再处理
        return fused_feature
    


        #return out           #返回增强特征
        
    def initialize(self):
        weight_init(self)

class IntensityPositionModule_flip(nn.Module):        #ICFE模块 rotate90°操作  调用SAM进行平均池化操作   修改
    '''multi-orientation intensity-based contrasted module (MIC)'''
    def __init__(self, inplanes, outplanes, g=1):
        super(IntensityPositionModule_flip, self).__init__()

        # self.SA1 = SAM(inplanes, outplanes, g)
        # self.SA2 = SAM(inplanes, outplanes, g)
        routing_function1 = RountingFunction_flip(in_channels=inplanes, kernel_number=3)
        self.conv = nn.Sequential(
                # RotatedBasicConv_k4(inplanes, inplanes, k=3, s=1, p=1, d=1, g=inplanes,rounting_func = routing_function1),
                RotatedBasicConv_flip(inplanes, inplanes, k=3, s=1, p=1, d=1, rounting_func = routing_function1, kernel_number=3),
                #basicConv(inplanes, outplanes, k=1, s=1, p=0, relu = True)
                basicConv(inplanes, outplanes, k=1, s=1, p=0, relu = True)
                )
    def forward(self, x):
        out = self.conv(x)
        return out           #返回增强特征[12, 64, 40, 40]
        
    def initialize(self):
        weight_init(self)

############################################## pooling #############################################
class PyramidPooling(nn.Module):   #Avg Pool        # PyramidPooling 实现了金字塔池化机制，通过对输入特征在不同尺度上进行自适应平均池化来提取多尺度特征。
    def __init__(self, in_channel, out_channel):       #池化（Pooling） 是卷积神经网络（Convolutional Neural Network, CNN）中一种重要的操作，用于减小特征图的尺寸，从而减少参数量，降低计算复杂度，同时也可以增强特征的平移不变性。池化操作通过对局部区域进行取值（如最大值或平均值），从而保留特征的主要信息，减轻数据的噪声影响。
        super(PyramidPooling, self).__init__()
        hidden_channel = int(in_channel / 4)
        self.conv1 = basicConv(in_channel, hidden_channel, k=1, s=1, p=0)
        self.conv2 = basicConv(in_channel, hidden_channel, k=1, s=1, p=0)
        self.conv3 = basicConv(in_channel, hidden_channel, k=1, s=1, p=0)
        self.conv4 = basicConv(in_channel, hidden_channel, k=1, s=1, p=0)
        self.out = basicConv(in_channel*2, out_channel, k=1, s=1, p=0)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = F.interpolate(self.conv1(F.adaptive_avg_pool2d(x, 1)), size)
        feat2 = F.interpolate(self.conv2(F.adaptive_avg_pool2d(x, 2)), size)
        feat3 = F.interpolate(self.conv3(F.adaptive_avg_pool2d(x, 3)), size)
        feat4 = F.interpolate(self.conv4(F.adaptive_avg_pool2d(x, 6)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)

        return x
        
    def initialize(self):
        weight_init(self)
##################################### net ###################################################

class AE(nn.Module):
    def __init__(self, N, C, in_channels, inter_channels, out_channels, pool=(2,2), factor=2):
        super().__init__()
        self.C = C
        self.trans = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU()
        )
        self.pool = nn.MaxPool2d(pool)
        self.linearKC = nn.Linear(inter_channels, C)
        self.linearNC = nn.Linear(N, C)
        self.gnn = GNN(inter_channels)
        self.spatialgnn = SpatialGNN(inter_channels, 24, 24)
        self.up = nn.Upsample(scale_factor=factor, mode='bilinear', align_corners=True)
        self.back = nn.Sequential(
            nn.Conv2d(inter_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, SP):
        '''x: bs, f, h, w
            SP: bs, C, h, w
        '''
        SP = torch.softmax(SP, dim=1)
        SP = self.pool(SP) ## bs, C, h/2, w/2
        SP = SP.reshape(SP.shape[0], SP.shape[1], -1) ## bs, C, n

        t = self.trans(x) ## bs, k, h, w
        y = self.pool(t) ## bs, k, h/2, w/2
        size = y.shape
        y = y.reshape(y.shape[0], y.shape[1], -1) ## bs, k, n
        sigma = self.linearKC(self.linearNC(y).permute(0, 2, 1)) ## bs, c, c
        A = torch.matmul(SP.permute(0, 2, 1), torch.matmul(sigma, SP)) ## bs, n, n

        y = y.permute(0, 2, 1) ## bs, n, k
        y = self.gnn(A, y) + y 
        y = self.spatialgnn(y) + y

        y = self.dropout(self.up( y.permute(0, 2, 1).reshape(size) )) + t
        y = self.back(y)
        return self.dropout(y)
    
    def initialize(self):
        # 遍历类的所有模块，并对权重进行初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)



class Net(nn.Module):       
    def __init__(self, cfg, backbone_path="./resnext/resnext_101_32x4d.pth"):
        super(Net, self).__init__()
        self.cfg = cfg
        # import pdb;pdb.set_trace()
        self.bkbone = ResNeXt101(backbone_path)         #GE
        # self.semantic_sidepath = R50MGSeg(num_classes = 25, pretrained=False, os=16)   #
        # self.AE = AE(24*24, 25, 64, 32, 32, pool=(1,1), factor=1) #初始化AE
        
        self.pyramid_pooling = PyramidPooling(2048, 64)
        self.conv1 = nn.ModuleList([
                basicConv(64, 64, k=1, s=1, p=0),
                basicConv(256, 64, k=1, s=1, p=0),
                basicConv(512, 64, k=1, s=1, p=0),
                basicConv(1024, 64, k=1, s=1, p=0),
                basicConv(2048, 64, k=1, s=1, p=0),
                basicConv(2048, 64, k=1, s=1, p=0)
                ])
        self.head = nn.ModuleList([
                conv3x3(64, 1, bias=True), #edge0
                conv3x3(64, 1, bias=True),
                conv3x3(64, 1, bias=True),
                conv3x3(64, 1, bias=True),
                conv3x3(64, 1, bias=True),
                conv3x3(64, 1, bias=True),
        ])
        self.ffm = nn.ModuleList([
                FFM(64),
                FFM(64),
                FFM(64),
                FFM(64),
                FFM(64)
        ])
        
        self.ipm = nn.ModuleList([
            IntensityPositionModule(64, 64),  #rotated
            IntensityPositionModule(64, 64),
            IntensityPositionModule(64, 64),
            IntensityPositionModule(64, 64),  #rotate
            IntensityPositionModule(64, 64)
        ])


        self.cam = CAM(64)

        self.ca1 = RFB_modified(1024, 64)
        self.ca2 = RFB_modified(2048, 64)

        self.refine = basicConv(64, 64, k=1, s=1, p=0)


        
        self.initialize()


    def forward(self, x, shape=None): 
        shape = x.size()[2:] if shape is None else shape
        # import pdb;pdb.set_trace()
        bk_stage1, bk_stage2, bk_stage3, bk_stage4, bk_stage5 = self.bkbone(x)
                
        fused4 = self.pyramid_pooling(bk_stage5)  #[12, 64, 7, 7]
        f5 = self.ca2(bk_stage5)   #[12, 64, 9, 9]
        # import pdb; pdb.set_trace()
        f5= self.ipm[3](f5)   #添加flip
        
        fused4 = F.interpolate(fused4, size=f5.size()[2:], mode='bilinear', align_corners=True) #[12, 64, 7, 7]
        fused3 = self.ffm[4](f5, fused4)

        f4 = self.ca1(bk_stage4)
        
        f4 = self.ipm[4](f4)    #添加flip
        
        fused3 = F.interpolate(fused3, size=f4.size()[2:], mode='bilinear', align_corners=True)
        fused2 = self.ffm[3](f4, fused3)

        f3 = self.conv1[2](bk_stage3)
        f3 = self.ipm[2](f3)
        
        f2 = self.conv1[1](bk_stage2)
        f2 = self.ipm[1](f2)
        
        
    
        f3 = F.interpolate(f3, size=f2.size()[2:], mode='bilinear', align_corners=True)
        fused1 = self.ffm[2](f2, f3)

        fused2 = F.interpolate(fused2, size=[fused1.size(2)//2, fused1.size(3)//2], mode='bilinear', align_corners=True)

        fused1 = self.cam(fused2, fused1)

        f1 = self.conv1[0](bk_stage1)
        f1 = self.ipm[0](f1)
        f2 = F.interpolate(f2, size=f1.size()[2:], mode='bilinear', align_corners=True)
        fused0 = self.ffm[1](f2, f1)

        fused1 = F.interpolate(fused1, size=fused0.size()[2:], mode='bilinear', align_corners=True)
        out0 = self.ffm[0](fused1, fused0)

        out0 = self.refine(out0)
        
        edge0 = F.interpolate(self.head[0](fused0), size=shape, mode='bilinear', align_corners=True)
        out0 = F.interpolate(self.head[1](out0), size=shape, mode='bilinear', align_corners=True)
        # import pdb;pdb.set_trace()
        if self.cfg.mode == 'train':
            out1 = F.interpolate(self.head[2](fused1), size=shape, mode='bilinear', align_corners=True)
            out2 = F.interpolate(self.head[3](fused2), size=shape, mode='bilinear', align_corners=True)
            out3 = F.interpolate(self.head[4](fused3), size=shape, mode='bilinear', align_corners=True)
            out4 = F.interpolate(self.head[5](fused4), size=shape, mode='bilinear', align_corners=True)
            return out0, edge0, out1, out2, out3, out4
        else:
            return out0, edge0

    def initialize(self):
        if self.cfg.snapshot:
            # print(f"Loading weights from: {self.cfg.snapshot}")
            self.load_state_dict(torch.load(self.cfg.snapshot))
        else:
            weight_init(self)