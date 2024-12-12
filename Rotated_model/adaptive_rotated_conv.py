import torch
import torch.nn as nn
from torch.nn import functional as F
from .routing_function import RountingFunction_flip,RountingFunction_flip_conv,RountingFunction_flip_conv_angle
# from GCN import AttenMultiHead
from sparse_attention import SparseAttention
# from Net_gan_GNN import FFM
from einops import rearrange, repeat
import cv2
import math
from timm.models.layers import trunc_normal_

__all__ = ['AdaptiveRotatedConv2d','AdaptiveRotatedConv2d_attention',]



# # Cross Aggregation Module
# class CAM(nn.Module):
#     def __init__(self, channel):
#         super(CAM, self).__init__()
#         self.down = nn.Sequential(
#             conv3x3(channel, channel, stride=2),
#             nn.BatchNorm2d(channel)
#         )
#         self.conv_1 = conv3x3(channel, channel)
#         self.bn_1 = nn.BatchNorm2d(channel)
#         self.conv_2 = conv3x3(channel, channel)
#         self.bn_2 = nn.BatchNorm2d(channel)
#         self.mul = FFM(channel)

#     def forward(self, x_high, x_low):
#         left_1 = x_low
#         left_2 = F.relu(self.down(x_low), inplace=True)
#         right_1 = F.interpolate(x_high, size=x_low.size()[2:], mode='bilinear', align_corners=True)
#         right_2 = x_high
#         left = F.relu(self.bn_1(self.conv_1(left_1 * right_1)), inplace=True)
#         right = F.relu(self.bn_2(self.conv_2(left_2 * right_2)), inplace=True)
#         right = F.interpolate(right, size=x_low.size()[2:], mode='bilinear', align_corners=True)
#         out = self.mul(left, right)
#         return out

#     def initialize(self):
#         weight_init(self)



class FFM_norelu(nn.Module):
    def __init__(self, channel):
        super(FFM_norelu, self).__init__()
        self.conv_1 = conv3x3(channel, channel)
        self.bn_1 = nn.BatchNorm2d(channel)
        self.conv_2 = conv3x3(channel, channel)
        self.bn_2 = nn.BatchNorm2d(channel)

    def forward(self, x_1, x_2):
        #out = torch.cat((x_1, x_2), dim=1)
        out1 =self.conv_1(x_1)
        out2 =self.conv_2(x_2)
        out = out1 + out2
        
        return out

    def initialize(self):
        weight_init(self)




def visualize(input,str):
    feature = torch.sum(input, dim=(0, 1))
    feature = abs(feature)
    cv2.imwrite(str,50.0*feature.cpu().numpy())
    return 0
############################################ attention ##############################################
class Attention(nn.Module):
    def __init__(self, d_model):
        super(Attention, self).__init__()
        self.scale = 1.0 / (d_model ** 0.5)  # 缩放因子

    def forward(self, Q, K, V):
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # 通过 softmax 计算权重
        weights = torch.softmax(scores, dim=-1)
        # 加权求和
        output = torch.matmul(weights, V)
        return output


def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, dilation=dilation, bias=bias)

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        # 线性变换，用于生成 Q, K, V 矩阵

        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.q_proj = nn.Linear(dim, dim)
        # self.q_proj = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1, bias=True)
        # 输出的线性变换
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        self.softmax = nn.Softmax(dim=-1)
        # nn.init.normal_(self.q_proj.weight,mean=0, std=0.00)  # 或者 init.xavier_normal_
        # nn.init.zeros_(self.q_proj.bias)  # 将偏置初始化为 0
        
        # nn.init.kaiming_uniform_(self.q_proj.weight, nonlinearity='relu')  # 针对 ReLU 激活函数
        # nn.init.kaiming_uniform_(self.k_proj.weight, nonlinearity='relu')  # 针对 ReLU 激活函数
        # nn.init.kaiming_uniform_(self.v_proj.weight, nonlinearity='relu')  # 针对 ReLU 激活函数
        
    def forward(self, x1, x2):
        # x1 是 Query，x2 是 Key 和 Value
        # import pdb;pdb.set_trace()
        B, T1, C = x1.shape  # x1 的形状: [batch_size, seq_len1, dim]
        _, T2, _ = x2.shape  # x2 的形状: [batch_size, seq_len2, dim]

        # 生成 Q, K, V 矩阵
        Q = self.q_proj(x2).view(B, T1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x1).view(B, T2, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x1).view(B, T2, self.num_heads, self.head_dim).transpose(1, 2)
        # import pdb;pdb.set_trace()
        # 计算注意力得分
        attn_scores = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = self.softmax(attn_scores)  # 注意力权重
        attn_weights = self.dropout(attn_weights)  # dropout 防止过拟合

        # 使用注意力权重加权值矩阵
        attn_output = attn_weights @ V
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T1, C)

        # 输出线性变换
        output = self.out_proj(attn_output)
        return output

#  Feature Fusion Module
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

class norelu(nn.Module):
    def __init__(self, channel):
        super(norelu, self).__init__()
        self.conv_1 = nn.Conv2d(channel, channel, kernel_size=3, stride=3, padding=1, dilation=1, bias=False)
        # # # self.bn_1 = nn.BatchNorm2d(channel)
        # self.conv_2 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        # self.bn_2 = nn.BatchNorm2d(channel)

    def forward(self, x_1, x_2):
        #out = torch.cat((x_1, x_2), dim=1)
        # out1 =self.conv_1(x_1)
        # out2 =self.conv_2(x_2)
        # out = out1 * out2
        
        return 0

    # def initialize(self):
    #     weight_init(self)



class FFM_3in(nn.Module):
    def __init__(self, channel):
        super(FFM_3in, self).__init__()
        self.conv_1 = conv3x3(channel, channel)
        self.bn_1 = nn.BatchNorm2d(channel)
        self.conv_2 = conv3x3(channel, channel)
        self.bn_2 = nn.BatchNorm2d(channel)
        # self.conv_3 = conv3x3(channel, channel)
        # self.bn_3 = nn.BatchNorm2d(channel)
    def forward(self, x_1, x_2,x_3):
        out = x_1 * x_2 * x_3
        #out = torch.cat((x_1, x_2), dim=1)
        out = F.relu(self.bn_1(self.conv_1(out)), inplace=True)
        out = F.relu(self.bn_2(self.conv_2(out)), inplace=True)
        return out

    def initialize(self):
        weight_init(self)


class FFM_3in_conv(nn.Module):
    def __init__(self, channel):
        super(FFM_3in_conv, self).__init__()
        self.conv_1 = conv3x3(channel, channel)
        self.bn_1 = nn.BatchNorm2d(channel)
        self.conv_2 = conv3x3(channel, channel)
        self.bn_2 = nn.BatchNorm2d(channel)
        self.conv_3 = conv3x3(channel, channel)
        self.bn_3 = nn.BatchNorm2d(channel)
    def forward(self, x_1, x_2,x_3):
        # out = x_1 * x_2
        #out = torch.cat((x_1, x_2), dim=1)
        out1 = F.relu(self.bn_1(self.conv_1(x_1)))
        out2 = F.relu(self.bn_2(self.conv_2(x_2)))
        out3 = F.relu(self.bn_3(self.conv_3(x_3)))
        out = out1 * out2 *out3
        return out

    def initialize(self):
        weight_init(self)
        
        

class FFM_conv(nn.Module):
    def __init__(self, channel):
        super(FFM_conv, self).__init__()
        self.conv_1 = conv3x3(channel, channel)
        self.bn_1 = nn.BatchNorm2d(channel)
        self.conv_2 = conv3x3(channel, channel)
        self.bn_2 = nn.BatchNorm2d(channel)

    def forward(self, x_1, x_2):
        # out = x_1 * x_2
        #out = torch.cat((x_1, x_2), dim=1)
        out1 = F.relu(self.bn_1(self.conv_1(x_1)))
        out2 = F.relu(self.bn_2(self.conv_2(x_2)))
        out = out1 * out2
        return out

    def initialize(self):
        weight_init(self)  

class FFM_conv_silu(nn.Module):
    def __init__(self, channel):
        super(FFM_conv_silu, self).__init__()
        self.conv_1 = conv3x3(channel, channel)
        self.bn_1 = nn.BatchNorm2d(channel)
        self.conv_2 = conv3x3(channel, channel)
        self.bn_2 = nn.BatchNorm2d(channel)

    def forward(self, x_1, x_2):
        # out = x_1 * x_2
        #out = torch.cat((x_1, x_2), dim=1)
        out1 = F.silu(self.bn_1(self.conv_1(x_1)))
        out2 = F.silu(self.bn_2(self.conv_2(x_2)))
        out = out1 * out2
        return out

    def initialize(self):
        weight_init(self)



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
            if hasattr(m, 'initialize'):
                m.initialize()            #通过 hasattr(m, 'initialize') 来检查模块是否有 initialize() 方法


def _get_flip_matrix(thetas):   #这段代码根据输入角度的正负性，选择正旋转矩阵或负旋转矩阵，生成一个适用于整个批次和组的旋转矩阵张量。
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


def _get_rotation_matrix(thetas):   #这段代码根据输入角度的正负性，选择正旋转矩阵或负旋转矩阵，生成一个适用于整个批次和组的旋转矩阵张量。
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
        batch_size = b
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



def batch_rotate_multiweight2(weights, lambdas, thetas):  #表示经过旋转后的卷积核权重
    #weights：形状：[kernel_number, Cout, Cin, k, k] kernel_number卷积核数量
    """
    Let
        batch_size = b
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


def batch_rotate_multiweight3(weights, lambdas, thetas):  #表示经过旋转后的卷积核权重
    #weights：形状：[kernel_number, Cout, Cin, k, k] kernel_number卷积核数量
    """
    Let
        batch_size = b
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




def batch_flip_multiweight(weights, lambdas, thetas):  #表示经过翻转后的卷积核权重
    #weights：形状：[kernel_number, Cout, Cin, k, k] kernel_number卷积核数量
    """
    Let
        batch_size = b
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
    # import pdb;pdb.set_trace()
    if k == 3 :
        # Stage 1:
        # input: thetas: [b, n]
        #        lambdas: [b, n]
        # output: rotation_matrix: [b, n, 9, 9] (with gate) --> [b*9, n*9]

        #       Sub_Stage 1.1:
        #       input: [b, n] kernel
        #       output: [b, n, 9, 9] rotation matrix
        rotation_matrix = _get_flip_matrix(thetas)

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



def batch_flip_multiweight2(weights, lambdas, thetas):  #表示经过翻转后的卷积核权重
    #weights：形状：[kernel_number, Cout, Cin, k, k] kernel_number卷积核数量
    """
    Let
        batch_size = b
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
    # import pdb;pdb.set_trace()
    if k == 3 :
        # Stage 1:
        # input: thetas: [b, n]
        #        lambdas: [b, n]
        # output: rotation_matrix: [b, n, 9, 9] (with gate) --> [b*9, n*9]

        #       Sub_Stage 1.1:
        #       input: [b, n] kernel
        #       output: [b, n, 9, 9] rotation matrix
        rotation_matrix = _get_flip_matrix(thetas)

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

def batch_flip_multiweight3(weights, lambdas, thetas):  #表示经过翻转后的卷积核权重
    #weights：形状：[kernel_number, Cout, Cin, k, k] kernel_number卷积核数量
    """
    Let
        batch_size = b
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
    # import pdb;pdb.set_trace()
    if k == 3 :
        # Stage 1:
        # input: thetas: [b, n]
        #        lambdas: [b, n]
        # output: rotation_matrix: [b, n, 9, 9] (with gate) --> [b*9, n*9]

        #       Sub_Stage 1.1:
        #       input: [b, n] kernel
        #       output: [b, n, 9, 9] rotation matrix
        rotation_matrix = _get_flip_matrix(thetas)

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



class GatingModule(nn.Module):
    def __init__(self,in_channels=64, channels=512):
        super().__init__()
        self.gpool = nn.AdaptiveAvgPool2d((1,1))
        self.trans_x = nn.Sequential(
            nn.Conv2d(in_channels, channels, 1),
            nn.BatchNorm2d(channels)
        )
        self.trans_y = nn.Sequential(
            nn.Conv2d(in_channels, channels, 1),
            nn.BatchNorm2d(channels)
        )        
        self.fc = nn.Sequential(
            nn.Linear(channels, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.scores = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x, y):
        bs, k, h, w = x.shape
        #import pdb; pdb.set_trace()
        
        #import pdb; pdb.set_trace()
        #bs, Cin, h, w = x.shape
        #x = x.reshape(1, bs * Cin, h, w)  # [1, bs * Cin, h, w]
        
        x = self.trans_x(x)#torch.Size([12, 64, 44, 44])
        y = self.trans_y(y)#torch.Size([12, 64, 44, 44])
        # import pdb;pdb.set_trace()
        fx = self.fc(self.gpool(x).reshape(bs, k))
        fy = self.fc(self.gpool(y).reshape(bs, k))
        scores = torch.softmax(self.scores( torch.cat([fx,fy],dim=1)), dim=-1) ## two weights
        score_x = scores[:, 0].reshape(bs, 1, 1, 1)
        score_y = scores[:, 1].reshape(bs, 1, 1, 1)
        # print("scores_x:",score_x.mean().item(), "scores_y:", score_y.mean().item())
        ret = score_x * x + score_y * y
        ret = self.conv(ret)
        return ret #, nn.L1Loss()(score_x, score_y) * 0.1
    
    def initialize(self):
        # 使用 weight_init 初始化子模块的权重
        weight_init(self)

# class AdaptiveRotatedConv2d_flip(nn.Module):    
#     def __init__(self, in_channels, out_channels, kernel_size,
#                 stride=1, padding=1, dilation=1, groups=1, bias=False,
#                 kernel_number=1, rounting_func=None, rotate_func=batch_rotate_multiweight):
#         super().__init__()
#         self.kernel_number = kernel_number
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.dilation = dilation
#         self.groups = groups
#         self.bias = bias
#         self.rounting_func = rounting_func
        
#         self.flip_func = batch_flip_multiweight
#         self.rotate_func = rotate_func

#         self.weight = nn.Parameter(
#             torch.Tensor(
#                 kernel_number, 
#                 out_channels,
#                 in_channels // groups,
#                 kernel_size,
#                 kernel_size,
#             )
#         )
        
#         self.weight_flip = nn.Parameter(
#             torch.Tensor(
#                 kernel_number, 
#                 out_channels,
#                 in_channels // groups,
#                 kernel_size,
#                 kernel_size,
#             )
#         )

        
#         # self.FFM = FFM_conv(in_channels)

#         #添加两层GCN学习rotated和flip
#         #self.gcn1 = GCNConv(out_channels, out_channels)
#         #self.gcn2 = GCNConv(out_channels, out_channels)
#         # self.cross_attention = CrossAttention(64)
#         # self.conv1=nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
#         # self.conv2=nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
#         nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
#         nn.init.kaiming_normal_(self.weight_flip, mode='fan_out', nonlinearity='relu')
#     def forward(self, x):
#         # get alphas, angles
#         # # [bs, Cin, h, w] --> [bs, n_theta], [bs, n_theta]
#         alphas, angles, alphas_flip = self.rounting_func(x)
#         angles_flip = angles.clone()
#         # rotate weight
#         # # [Cout, Cin, k, k] --> [bs * Cout, Cin, k, k]
#         # print(self.weight.shape)
#         # import pdb; pdb.set_trace()
#         rotated_weight = self.rotate_func(self.weight, alphas, angles)
#         flipped_weight = self.flip_func(self.weight_flip, alphas_flip, angles_flip)
#         rotated_weight_flipped = torch.flip(flipped_weight, dims=[-1])
#         # import pdb; pdb.set_trace()

#         # reshape images
#         bs, Cin, h, w = x.shape
#         x = x.reshape(1, bs * Cin, h, w)  # [1, bs * Cin, h, w]
        
#         # adaptive conv over images using group conv
#         out_rotated = F.conv2d(input=x, weight=rotated_weight, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=(self.groups * bs))
#         out_flipped = F.conv2d(input=x, weight=rotated_weight_flipped, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=(self.groups * bs))
#         #import pdb; pdb.set_trace()
        
    
        
#         # reshape back
#         out_rotated = out_rotated.reshape(bs, self.out_channels, *out_rotated.shape[2:])
#         out_flipped = out_flipped.reshape(bs, self.out_channels, *out_flipped.shape[2:])
        
#         #han_add_20241126
#         # import pdb;pdb.set_trace()
#         # out_rotated1 = self.conv1(out_rotated)
#         # out_flipped1 = self.conv2(out_flipped)
#         # import pdb;pdb.set_trace()
#         # out_rotated_pool = out_rotated.reshape(out_rotated.shape[0], out_rotated.shape[1], -1).permute(0, 2, 1) ## bs, n, k  [12, 256, 64]
#         # out_fliped_pool =  out_flipped.reshape( out_flipped .shape[0], out_flipped.shape[1], -1).permute(0, 2, 1)  #[12, 256, 64]
#         # out_attention= self.cross_attention(out_rotated_pool,out_fliped_pool)
#         # out_attention = out_attention.permute(0, 2, 1)           #[12, 64, 256]
#         # out_attention = out_attention.reshape(out_flipped.shape[0], self.out_channels, *out_flipped.shape[2:]) 
#         #han_end
        
#         # out = self.FFM(out_rotated, out_flipped) #使用 FFM 融合卷积特征
#         out = 0.7*out_rotated+ 0.3*out_flipped 
#         # import pdb; pdb.set_trace()
#         # import pdb; pdb.set_trace()
#         #return out
#         return out

        
#         # import pdb;pdb.set_trace()22
#         #x = 0.7*out_rotated + 0.3*out_flipped
#         #return x



#     def extra_repr(self):
#         s = ('{in_channels}, {out_channels}, kernel_number={kernel_number}'
#             ', kernel_size={kernel_size}, stride={stride}, bias={bias}')
        
#         if self.padding != (0,) * len([self.padding]):
#             s += ', padding={padding}'
#         if self.dilation != (1,) * len([self.dilation]):
#             s += ', dilation={dilation}'
#         if self.groups != 1:
#             s += ', groups={groups}'
#         return s.format(**self.__dict__)
    
#     def initialize(self):
#         weight_init(self)



class AdaptiveRotatedConv2d_flip(nn.Module):    
    def __init__(self, in_channels, out_channels, kernel_size,
                stride=1, padding=1, dilation=1, groups=1, bias=False,
                kernel_number=1, rounting_func=None, rotate_func=batch_rotate_multiweight):
        super().__init__()
        self.kernel_number = kernel_number
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.rounting_func = rounting_func
        
        self.flip_func = batch_flip_multiweight
        self.rotate_func = rotate_func

        self.weight = nn.Parameter(
            torch.Tensor(
                kernel_number, 
                out_channels,
                in_channels // groups,
                kernel_size,
                kernel_size,
            )
        )
        
        self.weight_flip = nn.Parameter(
            torch.Tensor(
                kernel_number, 
                out_channels,
                in_channels // groups,
                kernel_size,
                kernel_size,
            )
        )

        
        self.FFM = FFM_conv(in_channels)

        #添加两层GCN学习rotated和flip
        #self.gcn1 = GCNConv(out_channels, out_channels)
        #self.gcn2 = GCNConv(out_channels, out_channels)
        # self.cross_attention = CrossAttention(64)
        # self.conv1=nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # self.conv2=nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.weight_flip, mode='fan_out', nonlinearity='relu')
    def forward(self, x):
        # get alphas, angles
        # # [bs, Cin, h, w] --> [bs, n_theta], [bs, n_theta]
        alphas, angles, alphas_flip = self.rounting_func(x)
        # angles_flip = angles.clone()
        # rotate weight
        # # [Cout, Cin, k, k] --> [bs * Cout, Cin, k, k]
        # print(self.weight.shape)
        # import pdb; pdb.set_trace()
        rotated_weight = self.rotate_func(self.weight, alphas, angles)
        flipped_weight = self.flip_func(self.weight_flip, alphas_flip, alphas)
        rotated_weight_flipped = torch.flip(flipped_weight, dims=[-1])
        # import pdb; pdb.set_trace()

        # reshape images
        bs, Cin, h, w = x.shape
        x = x.reshape(1, bs * Cin, h, w)  # [1, bs * Cin, h, w]
        
        # adaptive conv over images using group conv
        out_rotated = F.conv2d(input=x, weight=rotated_weight, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=(self.groups * bs))
        out_flipped = F.conv2d(input=x, weight=rotated_weight_flipped, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=(self.groups * bs))
        #import pdb; pdb.set_trace()
        
    
        
        # reshape back
        out_rotated = out_rotated.reshape(bs, self.out_channels, *out_rotated.shape[2:])
        out_flipped = out_flipped.reshape(bs, self.out_channels, *out_flipped.shape[2:])
        
        #han_add_20241126
        # import pdb;pdb.set_trace()
        # out_rotated1 = self.conv1(out_rotated)
        # out_flipped1 = self.conv2(out_flipped)
        # import pdb;pdb.set_trace()
        # out_rotated_pool = out_rotated.reshape(out_rotated.shape[0], out_rotated.shape[1], -1).permute(0, 2, 1) ## bs, n, k  [12, 256, 64]
        # out_fliped_pool =  out_flipped.reshape( out_flipped .shape[0], out_flipped.shape[1], -1).permute(0, 2, 1)  #[12, 256, 64]
        # out_attention= self.cross_attention(out_rotated_pool,out_fliped_pool)
        # out_attention = out_attention.permute(0, 2, 1)           #[12, 64, 256]
        # out_attention = out_attention.reshape(out_flipped.shape[0], self.out_channels, *out_flipped.shape[2:]) 
        #han_end
        
        out = self.FFM(out_rotated, out_flipped) #使用 FFM 融合卷积特征
        # out = 0.7*out_rotated+ 0.3*out_flipped 

        #return out
        return out

        
        # import pdb;pdb.set_trace()22
        #x = 0.7*out_rotated + 0.3*out_flipped
        #return x



    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_number={kernel_number}'
            ', kernel_size={kernel_size}, stride={stride}, bias={bias}')
        
        if self.padding != (0,) * len([self.padding]):
            s += ', padding={padding}'
        if self.dilation != (1,) * len([self.dilation]):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        return s.format(**self.__dict__)
    
    def initialize(self):
        weight_init(self)


class AdaptiveRotatedConv2d_diliate_flip3(nn.Module):    
    def __init__(self, in_channels, out_channels, kernel_size,
                stride=1, padding=1, dilation=1, groups=1, bias=False,
                kernel_number=1, rounting_func=None, rotate_func=batch_rotate_multiweight):
        super().__init__()
        self.kernel_number = kernel_number
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.rounting_func = rounting_func

        # self.flip_func3 = batch_flip_multiweight3
        self.rotate_func = rotate_func
        self.rotate_func2 = batch_rotate_multiweight2
        self.rotate_func3 = batch_rotate_multiweight3
        self.flip_func = batch_flip_multiweight
        self.flip_func2 = batch_flip_multiweight2
        self.flip_func3 = batch_flip_multiweight3
        
        
        self.weight = nn.Parameter(
            torch.Tensor(
                kernel_number, 
                out_channels,
                in_channels // groups,
                kernel_size,
                kernel_size,
            )
        )
        
        self.weight2 = nn.Parameter(
            torch.Tensor(
                kernel_number, 
                out_channels,
                in_channels // groups,
                kernel_size,
                kernel_size,
            )
        )
        
        self.weight3 = nn.Parameter(
            torch.Tensor(
                kernel_number, 
                out_channels,
                in_channels // groups,
                kernel_size,
                kernel_size,
            )
        )
        
        self.weight_flip = nn.Parameter(
            torch.Tensor(
                kernel_number, 
                out_channels,
                in_channels // groups,
                kernel_size,
                kernel_size,
            )
        )

        self.weight_flip2 = nn.Parameter(
            torch.Tensor(
                kernel_number, 
                out_channels,
                in_channels // groups,
                kernel_size,
                kernel_size,
            )
        )
        
        self.weight_flip3 = nn.Parameter(
            torch.Tensor(
                kernel_number, 
                out_channels,
                in_channels // groups,
                kernel_size,
                kernel_size,
            )
        )
        
        
        self.FFM1 = FFM_conv(in_channels)
        self.FFM2 = FFM_conv(in_channels)
        self.FFM3 = FFM_conv(in_channels)
        # self.FFM_3in_1 = FFM_3in_conv(in_channels)
        # self.FFM_3in_2 = FFM_3in_conv(in_channels)
        # self.FFM3 = FFM(in_channels)
        #添加两层GCN学习rotated和flip
        #self.gcn1 = GCNConv(out_channels, out_channels)
        #self.gcn2 = GCNConv(out_channels, out_channels)
        # self.cross_attention = CrossAttention(64)
        # import pdb;pdb.set_trace()
        # self.conv1=nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # self.conv2=nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.weight, mode='fan_out',nonlinearity='relu')
        nn.init.kaiming_normal_(self.weight_flip, mode='fan_out',nonlinearity='relu')
        nn.init.kaiming_normal_(self.weight2, mode='fan_out',nonlinearity='relu')
        nn.init.kaiming_normal_(self.weight_flip2, mode='fan_out',nonlinearity='relu')
        nn.init.kaiming_normal_(self.weight3, mode='fan_out',nonlinearity='relu')
        nn.init.kaiming_normal_(self.weight_flip3, mode='fan_out',nonlinearity='relu')
    def forward(self, x):
        # get alphas, angles
        # # [bs, Cin, h, w] --> [bs, n_theta], [bs, n_theta]
        # import pdb; pdb.set_trace()
        alphas, angles, alphas_flip = self.rounting_func(x)
        angles_flip = angles.clone()
        # rotate weight
        # # [Cout, Cin, k, k] --> [bs * Cout, Cin, k, k]
        # print(self.weight.shape)
        # import pdb; pdb.set_trace()
        rotated_weight = self.rotate_func(self.weight, alphas, angles)
        rotated_weight2 = self.rotate_func2(self.weight2, alphas, angles)
        rotated_weight3 = self.rotate_func3(self.weight3, alphas, angles)
        
        
        flipped_weight = self.flip_func(self.weight_flip, alphas_flip, angles_flip)
        flipped_weight2 = self.flip_func2(self.weight_flip2, alphas_flip, angles_flip)
        flipped_weight3 = self.flip_func2(self.weight_flip3, alphas_flip, angles_flip)
        
        rotated_weight_flipped = torch.flip(flipped_weight, dims=[-1])
        rotated_weight_flipped2 = torch.flip(flipped_weight2, dims=[-1])
        rotated_weight_flipped3 = torch.flip(flipped_weight3, dims=[-1])
        # import pdb; pdb.set_trace()

        # reshape images
        bs, Cin, h, w = x.shape
        x = x.reshape(1, bs * Cin, h, w)  # [1, bs * Cin, h, w]
        
        # adaptive conv over images using group conv
        out_rotated = F.conv2d(input=x, weight=rotated_weight, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=(self.groups * bs))
        out_flipped = F.conv2d(input=x, weight=rotated_weight_flipped, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=(self.groups * bs))

        # import pdb; pdb.set_trace()
        out_rotated2 = F.conv2d(input=x, weight=rotated_weight2, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=(self.groups * bs))
        out_flipped2 = F.conv2d(input=x, weight=rotated_weight_flipped2, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=(self.groups * bs))
        
        out_rotated3 = F.conv2d(input=x, weight=rotated_weight3, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=(self.groups * bs))
        out_flipped3 = F.conv2d(input=x, weight=rotated_weight_flipped3, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=(self.groups * bs))

        
        # out_rotated2 = F.interpolate(out_rotated2, size=(out_rotated.shape[2], out_rotated.shape[3]), mode='bilinear', align_corners=False)
        # out_flipped2 = F.interpolate(out_flipped2, size=(out_rotated.shape[2], out_rotated.shape[3]), mode='bilinear', align_corners=False)

        # out_rotated3 = F.interpolate(out_rotated3, size=(out_rotated.shape[2], out_rotated.shape[3]), mode='bilinear', align_corners=False)
        # out_flipped3 = F.interpolate(out_flipped3, size=(out_rotated.shape[2], out_rotated.shape[3]), mode='bilinear', align_corners=False)


        # import pdb; pdb.set_trace()
        
        # reshape back
        out_rotated = out_rotated.reshape(bs, self.out_channels, *out_rotated.shape[2:])
        out_flipped = out_flipped.reshape(bs, self.out_channels, *out_flipped.shape[2:])
        out_rotated2 = out_rotated2.reshape(bs, self.out_channels, *out_rotated2.shape[2:])
        out_flipped2 = out_flipped2.reshape(bs, self.out_channels, *out_flipped2.shape[2:])
        
        out_rotated3 = out_rotated3.reshape(bs, self.out_channels, *out_rotated3.shape[2:])
        out_flipped3 = out_flipped3.reshape(bs, self.out_channels, *out_flipped3.shape[2:])
        
        out1 = self.FFM1(out_rotated, out_flipped)
        out2 = self.FFM2(out_rotated2, out_flipped2)
        out3 = self.FFM2(out_rotated3, out_flipped3)
        
        # out1 = self.FFM_3in_1(out_rotated, out_rotated2,out_rotated3)
        # out2 = self.FFM_3in_2(out_flipped, out_flipped2,out_flipped3)
        # out3 = self.FFM2(out_rotated3, out_flipped3)
        
        
        out = 0.5*out1+ 0.25*out2 + 0.25*out3
 
        # out_flipped = out_flipped + out_flipped2

        # out = self.FFM(out_rotated, out_flipped) #使用 FFM 融合卷积特征
        
        # out = 0.7*out_rotated + 0.3*out_flipped
        # out = out_rotated + out
        # out = 0.7*out_rotated+ 0.3*out_flipped 
        # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        # #return out
        # visualize(out, '1_x_input.jpg')
        # visualize(out1, '1_out1.jpg')
        # visualize(out2, '1_out2.jpg')
        return out

        
        # import pdb;pdb.set_trace()22
        #x = 0.7*out_rotated + 0.3*out_flipped
        #return x

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_number={kernel_number}'
            ', kernel_size={kernel_size}, stride={stride}, bias={bias}')
        
        if self.padding != (0,) * len([self.padding]):
            s += ', padding={padding}'
        if self.dilation != (1,) * len([self.dilation]):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        return s.format(**self.__dict__)
    
    def initialize(self):
        weight_init(self)





# class AdaptiveRotatedConv2d_diliate_flip2(nn.Module):    
#     def __init__(self, in_channels, out_channels, kernel_size,
#                 stride=1, padding=1, dilation=1, groups=1, bias=False,
#                 kernel_number=1, rounting_func=None, rotate_func=batch_rotate_multiweight):
#         super().__init__()
#         self.kernel_number = kernel_number
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.dilation = dilation
#         self.groups = groups
#         self.bias = bias
#         self.rounting_func = rounting_func

#         # self.flip_func3 = batch_flip_multiweight3
#         self.rotate_func = rotate_func
#         self.rotate_func2 = batch_rotate_multiweight2
#         # self.rotate_func3 = batch_rotate_multiweight3
#         self.flip_func = batch_flip_multiweight
#         self.flip_func2 = batch_flip_multiweight2
#         # self.flip_func3 = batch_flip_multiweight3
        
        
#         self.weight = nn.Parameter(
#             torch.Tensor(
#                 kernel_number, 
#                 out_channels,
#                 in_channels // groups,
#                 kernel_size,
#                 kernel_size,
#             )
#         )
        
#         self.weight2 = nn.Parameter(
#             torch.Tensor(
#                 kernel_number, 
#                 out_channels,
#                 in_channels // groups,
#                 kernel_size,
#                 kernel_size,
#             )
#         )
    
        
#         self.weight_flip = nn.Parameter(
#             torch.Tensor(
#                 kernel_number, 
#                 out_channels,
#                 in_channels // groups,
#                 kernel_size,
#                 kernel_size,
#             )
#         )

#         self.weight_flip2 = nn.Parameter(
#             torch.Tensor(
#                 kernel_number, 
#                 out_channels,
#                 in_channels // groups,
#                 kernel_size,
#                 kernel_size,
#             )
#         )

#         self.FFM1 = FFM_norelu(in_channels)
#         self.FFM2 = FFM_norelu(in_channels)
#         self.FFM3 = FFM_conv(in_channels)
#         #添加两层GCN学习rotated和flip
#         #self.gcn1 = GCNConv(out_channels, out_channels)
#         #self.gcn2 = GCNConv(out_channels, out_channels)
#         # self.cross_attention = CrossAttention(64)

#         # self.conv1=nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
#         # self.conv2=nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
#         nn.init.kaiming_normal_(self.weight, mode='fan_out',nonlinearity='relu')
#         nn.init.kaiming_normal_(self.weight_flip, mode='fan_out',nonlinearity='relu')
#         nn.init.kaiming_normal_(self.weight2, mode='fan_out',nonlinearity='relu')
#         nn.init.kaiming_normal_(self.weight_flip2, mode='fan_out',nonlinearity='relu')
#         # nn.init.kaiming_normal_(self.weight3, mode='fan_out')
#         # nn.init.kaiming_normal_(self.weight_flip3, mode='fan_out')
#     def forward(self, x):
#         # get alphas, angles
#         # # [bs, Cin, h, w] --> [bs, n_theta], [bs, n_theta]
#         # import pdb; pdb.set_trace()
#         alphas, angles, alphas_flip = self.rounting_func(x)
#         angles_flip = angles.clone()
#         # import pdb; pdb.set_trace()
#         # rotate weight
#         # # [Cout, Cin, k, k] --> [bs * Cout, Cin, k, k]
#         # print(self.weight.shape)
#         # import pdb; pdb.set_trace()
#         rotated_weight = self.rotate_func(self.weight, alphas, angles)
#         rotated_weight2 = self.rotate_func2(self.weight2, alphas, angles)
#         # rotated_weight3 = self.rotate_func3(self.weight3, alphas, angles)
        
        
#         flipped_weight = self.flip_func(self.weight_flip, alphas_flip, angles_flip)
#         flipped_weight2 = self.flip_func2(self.weight_flip2, alphas_flip, angles_flip)
#         # flipped_weight3 = self.flip_func2(self.weight_flip3, alphas_flip, angles_flip)
        
#         rotated_weight_flipped = torch.flip(flipped_weight, dims=[-1])
#         rotated_weight_flipped2 = torch.flip(flipped_weight2, dims=[-1])
#         # rotated_weight_flipped3 = torch.flip(flipped_weight3, dims=[-1])
#         # import pdb; pdb.set_trace()

#         # reshape images
#         bs, Cin, h, w = x.shape
#         x = x.reshape(1, bs * Cin, h, w)  # [1, bs * Cin, h, w]
        
#         # adaptive conv over images using group conv
#         # import pdb;pdb.set_trace()
#         out_rotated = F.conv2d(input=x, weight=rotated_weight, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=(self.groups * bs))
#         out_flipped = F.conv2d(input=x, weight=rotated_weight_flipped, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=(self.groups * bs))

#         # import pdb; pdb.set_trace()
#         out_rotated2 = F.conv2d(input=x, weight=rotated_weight2, bias=None, stride=self.stride, padding=self.padding, dilation=2, groups=(self.groups * bs))
#         out_flipped2 = F.conv2d(input=x, weight=rotated_weight_flipped2, bias=None, stride=self.stride, padding=self.padding, dilation=2, groups=(self.groups * bs))
        
#         # out_rotated3 = F.conv2d(input=x, weight=rotated_weight3, bias=None, stride=self.stride, padding=self.padding, dilation=3, groups=(self.groups * bs))
#         # out_flipped3 = F.conv2d(input=x, weight=rotated_weight_flipped3, bias=None, stride=self.stride, padding=self.padding, dilation=3, groups=(self.groups * bs))

        
#         out_rotated2 = F.interpolate(out_rotated2, size=(out_rotated.shape[2], out_rotated.shape[3]), mode='bilinear', align_corners=False)
#         out_flipped2 = F.interpolate(out_flipped2, size=(out_rotated.shape[2], out_rotated.shape[3]), mode='bilinear', align_corners=False)

#         # out_rotated3 = F.interpolate(out_rotated3, size=(out_rotated.shape[2], out_rotated.shape[3]), mode='bilinear', align_corners=False)
#         # out_flipped3 = F.interpolate(out_flipped3, size=(out_rotated.shape[2], out_rotated.shape[3]), mode='bilinear', align_corners=False)


#         # import pdb; pdb.set_trace()
        
#         # reshape back
#         out_rotated = out_rotated.reshape(bs, self.out_channels, *out_rotated.shape[2:])
#         out_flipped = out_flipped.reshape(bs, self.out_channels, *out_flipped.shape[2:])
#         out_rotated2 = out_rotated2.reshape(bs, self.out_channels, *out_rotated2.shape[2:])
#         out_flipped2 = out_flipped2.reshape(bs, self.out_channels, *out_flipped2.shape[2:])
        

#         out1 = self.FFM1(out_rotated, out_rotated2)
#         out2 = self.FFM2(out_flipped, out_flipped2)
        
#         out = self.FFM3(out1, out2)
        
#         # out = 0.7*out1+ 0.3*out2
#         # out_rotated = out_rotated + out_rotated2
#         # out_flipped = out_flipped + out_flipped2

#         # out = self.FFM(out_rotated, out_flipped) #使用 FFM 融合卷积特征
        
#         # out = 0.7*out_rotated + 0.3*out_flipped
#         # out = out_rotated + out
#         # out = 0.7*out_rotated+ 0.3*out_flipped 
#         # import pdb; pdb.set_trace()
#         # import pdb; pdb.set_trace()
#         #return out
#         return out

        
#         # import pdb;pdb.set_trace()22
#         #x = 0.7*out_rotated + 0.3*out_flipped
#         #return x

#     def extra_repr(self):
#         s = ('{in_channels}, {out_channels}, kernel_number={kernel_number}'
#             ', kernel_size={kernel_size}, stride={stride}, bias={bias}')
        
#         if self.padding != (0,) * len([self.padding]):
#             s += ', padding={padding}'
#         if self.dilation != (1,) * len([self.dilation]):
#             s += ', dilation={dilation}'
#         if self.groups != 1:
#             s += ', groups={groups}'
#         return s.format(**self.__dict__)
    
#     def initialize(self):
#         weight_init(self)

class SAttention(nn.Module):
    def __init__(self, dim, sa_num_heads=8, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()

        self.dim = dim
        self.sa_num_heads = sa_num_heads

        assert dim % sa_num_heads == 0, f"dim {dim} should be divided by num_heads {sa_num_heads}."

        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        head_dim = dim // sa_num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape

        q = self.q(x).reshape(B, N, self.sa_num_heads, C // self.sa_num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, -1, 2, self.sa_num_heads, C // self.sa_num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C) + \
            self.local_conv(v.transpose(1, 2).reshape(B, N, C).transpose(1, 2).view(B, C, H, W)).view(B, C,
                                                                                                      N).transpose(1, 2)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x.permute(0, 2, 1).reshape(B, C, H, W)


# Conv_One_Identity
class COI(nn.Module):
    def __init__(self, inc, k=3, p=1):
        super().__init__()
        self.outc = inc
        self.dw = nn.Conv2d(inc, self.outc, kernel_size=k, padding=p, groups=inc)
        self.conv1_1 = nn.Conv2d(inc, self.outc, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(self.outc)
        self.bn2 = nn.BatchNorm2d(self.outc)
        self.bn3 = nn.BatchNorm2d(self.outc)
        self.act = nn.GELU()
        self.apply(self._init_weights)

    def forward(self, x):
        shortcut = self.bn1(x)

        x_dw = self.bn2(self.dw(x))
        x_conv1_1 = self.bn3(self.conv1_1(x))
        return self.act(shortcut + x_dw + x_conv1_1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


class MHMC(nn.Module):
    def __init__(self, dim, ca_num_heads=4, qkv_bias=True, proj_drop=0., ca_attention=1, expand_ratio=2):
        super().__init__()

        self.ca_attention = ca_attention
        self.dim = dim
        self.ca_num_heads = ca_num_heads

        assert dim % ca_num_heads == 0, f"dim {dim} should be divided by num_heads {ca_num_heads}."

        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.split_groups = self.dim // ca_num_heads

        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.s = nn.Linear(dim, dim, bias=qkv_bias)
        for i in range(self.ca_num_heads):
            local_conv = nn.Conv2d(dim // self.ca_num_heads, dim // self.ca_num_heads, kernel_size=(3 + i * 2),
                                   padding=(1 + i), stride=1,
                                   groups=dim // self.ca_num_heads)  # kernel_size 3,5,7,9 大核dw卷积，padding 1,2,3,4
            setattr(self, f"local_conv_{i + 1}", local_conv)
        self.proj0 = nn.Conv2d(dim, dim * expand_ratio, kernel_size=1, padding=0, stride=1,
                               groups=self.split_groups)
        self.bn = nn.BatchNorm2d(dim * expand_ratio)
        self.proj1 = nn.Conv2d(dim * expand_ratio, dim, kernel_size=1, padding=0, stride=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        v = self.v(x)
        s = self.s(x).reshape(B, H, W, self.ca_num_heads, C // self.ca_num_heads).permute(3, 0, 4, 1,
                                                                                          2)  # num_heads,B,C,H,W
        for i in range(self.ca_num_heads):
            local_conv = getattr(self, f"local_conv_{i + 1}")
            s_i = s[i]  # B,C,H,W
            s_i = local_conv(s_i).reshape(B, self.split_groups, -1, H, W)
            if i == 0:
                s_out = s_i
            else:
                s_out = torch.cat([s_out, s_i], 2)
        s_out = s_out.reshape(B, C, H, W)
        s_out = self.proj1(self.act(self.bn(self.proj0(s_out))))
        self.modulator = s_out
        s_out = s_out.reshape(B, C, N).permute(0, 2, 1)
        x = s_out * v

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MAFM(nn.Module):
    def __init__(self, inc):
        super().__init__()
        self.outc = inc
        self.attention = MHMC(dim=inc)
        self.coi = COI(inc)
        self.pw = nn.Sequential(
            nn.Conv2d(in_channels=inc, out_channels=inc, kernel_size=1, stride=1),
            nn.BatchNorm2d(inc),
            nn.GELU()
        )
        self.pre_att = nn.Sequential(
            nn.Conv2d(inc * 2, inc * 2, kernel_size=3, padding=1, groups=inc * 2),
            nn.BatchNorm2d(inc * 2),
            nn.GELU(),
            nn.Conv2d(inc * 2, inc, kernel_size=1),
            nn.BatchNorm2d(inc),
            nn.GELU()
        )

        self.apply(self._init_weights)

    def forward(self, x, d):
        # multi = x * d
        # B, C, H, W = x.shape
        # x_cat = torch.cat((x, d, multi), dim=1)

        B, C, H, W = x.shape
        x_cat = torch.cat((x, d), dim=1)
        x_pre = self.pre_att(x_cat)
        # Attention
        x_reshape = x_pre.flatten(2).permute(0, 2, 1)  # B,C,H,W to B,N,C
        attention = self.attention(x_reshape, H, W)  # attention
        attention = attention.permute(0, 2, 1).reshape(B, C, H, W)  # B,N,C to B,C,H,W

        # COI
        x_conv = self.coi(attention)  # dw3*3,1*1,identity
        x_conv = self.pw(x_conv)  # pw

        return x_conv

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


class DWPWConv(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=inc, out_channels=inc, kernel_size=3, padding=1, stride=1, groups=inc),
            nn.BatchNorm2d(inc),
            nn.GELU(),
            nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, stride=1),
            nn.BatchNorm2d(outc),
            nn.GELU()
        )

    def forward(self, x):
        return self.conv(x)
class GFM(nn.Module):
    def __init__(self, inc, expend_ratio=2):
        super().__init__()
        self.expend_ratio = expend_ratio
        assert expend_ratio in [2, 3], f"expend_ratio {expend_ratio} mismatch"

        self.sa = SAttention(dim=inc)
        self.dw_pw = DWPWConv(inc * expend_ratio, inc)
        self.act = nn.GELU()
        self.apply(self._init_weights)
    
    def forward(self, x, d):
        B, C, H, W = x.shape
        if self.expend_ratio == 2:
            cat = torch.cat((x, d), dim=1)
        else:
            multi = x * d
            cat = torch.cat((x, d, multi), dim=1)
        x_rc = self.dw_pw(cat).flatten(2).permute(0, 2, 1)
        # import pdb;pdb.set_trace()
        x_ = self.sa(x_rc, H, W)
        x_ = x_ + x
        return self.act(x_)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()



class AdaptiveRotatedConv2d_diliate_flip2(nn.Module):    
    def __init__(self, in_channels, out_channels, kernel_size,
                stride=1, padding=1, dilation=1, groups=1, bias=False,
                kernel_number=1, rounting_func=None, rotate_func=batch_rotate_multiweight):
        super().__init__()
        self.kernel_number = kernel_number
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.rounting_func = rounting_func

        # self.flip_func3 = batch_flip_multiweight3
        self.rotate_func = rotate_func
        self.rotate_func2 = batch_rotate_multiweight2
        # self.rotate_func3 = batch_rotate_multiweight3
        self.flip_func = batch_flip_multiweight
        self.flip_func2 = batch_flip_multiweight2
        # self.flip_func3 = batch_flip_multiweight3
        
        
        self.weight = nn.Parameter(
            torch.Tensor(
                kernel_number, 
                out_channels,
                in_channels // groups,
                kernel_size,
                kernel_size,
            )
        )
        
        self.weight2 = nn.Parameter(
            torch.Tensor(
                kernel_number, 
                out_channels,
                in_channels // groups,
                kernel_size,
                kernel_size,
            )
        )
    
        
        self.weight_flip = nn.Parameter(
            torch.Tensor(
                kernel_number, 
                out_channels,
                in_channels // groups,
                kernel_size,
                kernel_size,
            )
        )

        self.weight_flip2 = nn.Parameter(
            torch.Tensor(
                kernel_number, 
                out_channels,
                in_channels // groups,
                kernel_size,
                kernel_size,
            )
        )

        self.FFM1 = FFM_norelu(in_channels)
        self.FFM2 = FFM_norelu(in_channels)
        # self.FFM3 = FFM_conv(in_channels)
        self.mafm2 = MAFM(inc=in_channels)
        # self.gfm1 = GFM(inc=in_channels, expend_ratio=3)
        #添加两层GCN学习rotated和flip
        #self.gcn1 = GCNConv(out_channels, out_channels)
        #self.gcn2 = GCNConv(out_channels, out_channels)
        # self.cross_attention = CrossAttention(64)

        # self.conv1=nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # self.conv2=nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.weight, mode='fan_out',nonlinearity='relu')
        nn.init.kaiming_normal_(self.weight_flip, mode='fan_out',nonlinearity='relu')
        nn.init.kaiming_normal_(self.weight2, mode='fan_out',nonlinearity='relu')
        nn.init.kaiming_normal_(self.weight_flip2, mode='fan_out',nonlinearity='relu')
        # nn.init.kaiming_normal_(self.weight3, mode='fan_out')
        # nn.init.kaiming_normal_(self.weight_flip3, mode='fan_out')
    def forward(self, x):
        # get alphas, angles
        
        x1 = x
        # # [bs, Cin, h, w] --> [bs, n_theta], [bs, n_theta]
        # import pdb; pdb.set_trace()
        alphas, angles, alphas_flip,angles_flip = self.rounting_func(x)
        # angles_flip = angles.clone()
        # import pdb; pdb.set_trace()
        # rotate weight
        # # [Cout, Cin, k, k] --> [bs * Cout, Cin, k, k]
        # print(self.weight.shape)
        # import pdb; pdb.set_trace()
        rotated_weight = self.rotate_func(self.weight, alphas, angles)
        rotated_weight2 = self.rotate_func2(self.weight2, alphas, angles)
        # rotated_weight3 = self.rotate_func3(self.weight3, alphas, angles)
        
        # with torch.no_grad():
        flipped_weight = torch.flip(self.weight, dims=[-1])
        flipped_weight2 = torch.flip(self.weight2, dims=[-1])
        rotated_weight_flipped = self.flip_func(flipped_weight, alphas_flip, angles_flip)
        rotated_weight_flipped2 = self.flip_func2(flipped_weight2, alphas_flip, angles_flip)
        # flipped_weight3 = self.flip_func2(self.weight_flip3, alphas_flip, angles_flip)
        
        # rotated_weight_flipped = torch.flip(flipped_weight, dims=[-1])
        # rotated_weight_flipped2 = torch.flip(flipped_weight2, dims=[-1])
        # rotated_weight_flipped3 = torch.flip(flipped_weight3, dims=[-1])
        # import pdb; pdb.set_trace()

        # reshape images
        bs, Cin, h, w = x.shape
        x = x.reshape(1, bs * Cin, h, w)  # [1, bs * Cin, h, w]
        
        # adaptive conv over images using group conv
        # import pdb;pdb.set_trace()
        out_rotated = F.conv2d(input=x, weight=rotated_weight, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=(self.groups * bs))
        # out_rotated2 = F.conv2d(input=x, weight=rotated_weight2, bias=None, stride=self.stride, padding=self.padding, dilation=2, groups=(self.groups * bs))
        
        # with torch.no_grad():
        out_flipped = F.conv2d(input=x, weight=rotated_weight_flipped, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=(self.groups * bs))
        # out_flipped2 = F.conv2d(input=x, weight=rotated_weight_flipped2, bias=None, stride=self.stride, padding=self.padding, dilation=2, groups=(self.groups * bs))
        
        # out_rotated3 = F.conv2d(input=x, weight=rotated_weight3, bias=None, stride=self.stride, padding=self.padding, dilation=3, groups=(self.groups * bs))
        # out_flipped3 = F.conv2d(input=x, weight=rotated_weight_flipped3, bias=None, stride=self.stride, padding=self.padding, dilation=3, groups=(self.groups * bs))

        
        # out_rotated2 = F.interpolate(out_rotated2, size=(out_rotated.shape[2], out_rotated.shape[3]), mode='bilinear', align_corners=False)
        # out_flipped2 = F.interpolate(out_flipped2, size=(out_rotated.shape[2], out_rotated.shape[3]), mode='bilinear', align_corners=False)

        # out_rotated3 = F.interpolate(out_rotated3, size=(out_rotated.shape[2], out_rotated.shape[3]), mode='bilinear', align_corners=False)
        # out_flipped3 = F.interpolate(out_flipped3, size=(out_rotated.shape[2], out_rotated.shape[3]), mode='bilinear', align_corners=False)
        # import pdb; pdb.set_trace()
        
        # reshape back
        out_rotated = out_rotated.reshape(bs, self.out_channels, *out_rotated.shape[2:])
        out_flipped = out_flipped.reshape(bs, self.out_channels, *out_flipped.shape[2:])
        # out_rotated2 = out_rotated2.reshape(bs, self.out_channels, *out_rotated2.shape[2:])
        # out_flipped2 = out_flipped2.reshape(bs, self.out_channels, *out_flipped2.shape[2:])
        

        # out1 = self.FFM1(out_rotated, out_rotated2)
        # out2 = self.FFM2(out_flipped, out_flipped2)
        # out = self.FFM3(out1, out2)
        # x1 = x1 + out
        # import pdb;pdb.set_trace()
        # out = self.gfm1(out1, out2)
        out = self.mafm2(out_rotated, out_flipped)
        # out = 0.7*out1+ 0.3*out2
        # out_rotated = out_rotated + out_rotated2
        # out_flipped = out_flipped + out_flipped2

        # out = self.FFM(out_rotated, out_flipped) #使用 FFM 融合卷积特征
        
        # out = 0.7*out_rotated + 0.3*out_flipped
        # out = out_rotated + out
        # out = 0.7*out_rotated+ 0.3*out_flipped 
        # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        #return out
        return out

        
        # import pdb;pdb.set_trace()22
        #x = 0.7*out_rotated + 0.3*out_flipped
        #return x

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_number={kernel_number}'
            ', kernel_size={kernel_size}, stride={stride}, bias={bias}')
        
        if self.padding != (0,) * len([self.padding]):
            s += ', padding={padding}'
        if self.dilation != (1,) * len([self.dilation]):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        return s.format(**self.__dict__)
    
    def initialize(self):
        weight_init(self)



class AdaptiveRotatedConv2d_fattn_han(nn.Module):    
    def __init__(self, in_channels, out_channels, kernel_size,
                stride=1, padding=1, dilation=1, groups=1, bias=False,
                kernel_number=1, rounting_func=None, rotate_func=batch_rotate_multiweight):
        super().__init__()
        self.kernel_number = kernel_number
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.rounting_func = rounting_func
        
        self.flip_func = batch_flip_multiweight
        self.rotate_func = rotate_func

        self.weight = nn.Parameter(
            torch.Tensor(
                kernel_number, 
                out_channels,
                in_channels // groups,
                kernel_size,
                kernel_size,
            )
        )
        
        self.weight_flip = nn.Parameter(
            torch.Tensor(
                kernel_number, 
                out_channels,
                in_channels // groups,
                kernel_size,
                kernel_size,
            )
        )

        
        self.FFM = FFM_conv(in_channels)

        #添加两层GCN学习rotated和flip
        #self.gcn1 = GCNConv(out_channels, out_channels)
        #self.gcn2 = GCNConv(out_channels, out_channels)
        # self.cross_attention = CrossAttention(64)
        # import pdb;pdb.set_trace()
        # self.conv1=nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # self.conv2=nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.weight_flip, mode='fan_out', nonlinearity='relu')
    def forward(self, x):
        # get alphas, angles
        # # [bs, Cin, h, w] --> [bs, n_theta], [bs, n_theta]
        alphas, angles, alphas_flip = self.rounting_func(x)
        angles_flip = angles.clone()
        # rotate weight
        # # [Cout, Cin, k, k] --> [bs * Cout, Cin, k, k]
        # print(self.weight.shape)
        # import pdb; pdb.set_trace()
        rotated_weight = self.rotate_func(self.weight, alphas, angles)
        flipped_weight = self.flip_func(self.weight_flip, alphas_flip, angles_flip)
        rotated_weight_flipped = torch.flip(flipped_weight, dims=[-1])
        # import pdb; pdb.set_trace()

        # reshape images
        bs, Cin, h, w = x.shape
        x = x.reshape(1, bs * Cin, h, w)  # [1, bs * Cin, h, w]
        
        # adaptive conv over images using group conv
        out_rotated = F.conv2d(input=x, weight=rotated_weight, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=(self.groups * bs))
        out_flipped = F.conv2d(input=x, weight=rotated_weight_flipped, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=(self.groups * bs))
        #import pdb; pdb.set_trace()
        
    
        
        # reshape back
        out_rotated = out_rotated.reshape(bs, self.out_channels, *out_rotated.shape[2:])
        out_flipped = out_flipped.reshape(bs, self.out_channels, *out_flipped.shape[2:])
        
        #han_add_20241126
        # import pdb;pdb.set_trace()
        # out_rotated1 = self.conv1(out_rotated)
        # out_flipped1 = self.conv2(out_flipped)
        # import pdb;pdb.set_trace()
        # out_rotated_pool = out_rotated.reshape(out_rotated.shape[0], out_rotated.shape[1], -1).permute(0, 2, 1) ## bs, n, k  [12, 256, 64]
        # out_fliped_pool =  out_flipped.reshape( out_flipped .shape[0], out_flipped.shape[1], -1).permute(0, 2, 1)  #[12, 256, 64]
        # out_attention= self.cross_attention(out_rotated_pool,out_fliped_pool)
        # out_attention = out_attention.permute(0, 2, 1)           #[12, 64, 256]
        # out_attention = out_attention.reshape(out_flipped.shape[0], self.out_channels, *out_flipped.shape[2:]) 
        #han_end
        
        out = self.FFM(out_rotated, out_flipped) #使用 FFM 融合卷积特征
        
        # out = out_rotated + out
        # out = 0.7*out_rotated+ 0.3*out_flipped 
        # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        #return out
        return out

        
        # import pdb;pdb.set_trace()22
        #x = 0.7*out_rotated + 0.3*out_flipped
        #return x



    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_number={kernel_number}'
            ', kernel_size={kernel_size}, stride={stride}, bias={bias}')
        
        if self.padding != (0,) * len([self.padding]):
            s += ', padding={padding}'
        if self.dilation != (1,) * len([self.dilation]):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        return s.format(**self.__dict__)
    
    def initialize(self):
        weight_init(self)




class AdaptiveRotatedConv2d_fusion(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                stride=1, padding=1, dilation=1, groups=1, bias=False,
                kernel_number=1, rounting_func=None, rotate_func=batch_rotate_multiweight):
        super().__init__()
        self.kernel_number = kernel_number
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.rounting_func = rounting_func
        self.rotate_func = rotate_func
        self.weight = nn.Parameter(
            torch.Tensor(
                kernel_number, 
                out_channels,
                in_channels // groups,
                kernel_size,
                kernel_size,
            )
        )
        
        # 定义旋转函数
        #self.routing_func = rounting_func
        #self.rotate_func = rotate_func
        
        # 添加 GatingModule
        #self.gating_module = GatingModule(in_channels=64, channels=out_channels)
        self.FFM = FFM(in_channels)
        

        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        alphas, angles = self.rounting_func(x)       # shape:12 个批次的 1D 张量
        rotated_weight = self.rotate_func(self.weight, alphas, angles)#生成“旋转后的卷积核权重
        rotated_weight_flipped = torch.flip(rotated_weight, dims=[-1])#对卷积核翻转
        #import pdb; pdb.set_trace()
        
        
        bs, Cin, h, w = x.shape
        x = x.reshape(1, bs * Cin, h, w)  # [1, bs * Cin, h, w]

        conv_rotated = F.conv2d(input=x, weight=rotated_weight, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation,groups=(self.groups * bs))#torch.Size([1, 768, 32, 32])

        conv_flipped = F.conv2d(input=x, weight=rotated_weight_flipped , bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation,groups=(self.groups * bs))#torch.Size([1, 768, 32, 32])n


        conv_rotated = conv_rotated.reshape(bs, self.out_channels, *conv_rotated.shape[2:])#torch.Size([12, 64, 32, 32])
        conv_flipped = conv_flipped.reshape(bs, self.out_channels, *conv_flipped.shape[2:])#torch.Size([12, 64, 32, 32])

        fused_feature = self.FFM(conv_rotated, conv_flipped) #使用 FFM 融合卷积特征
        # import pdb; pdb.set_trace()
        #return out
        return fused_feature

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_number={kernel_number}'
            ', kernel_size={kernel_size}, stride={stride}, bias={bias}')
        
        if self.padding != (0,) * len([self.padding]):
            s += ', padding={padding}'
            
        if self.dilation != (1,) * len([self.dilation]):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        return s.format(**self.__dict__)
    
    def initialize(self):
        weight_init(self)  

class AdaptiveRotatedConv2d(nn.Module):    
    def __init__(self, in_channels, out_channels, kernel_size,
                stride=1, padding=1, dilation=1, groups=1, bias=False,
                kernel_number=1, rounting_func=None, rotate_func=batch_rotate_multiweight):
        super().__init__()
        self.kernel_number = kernel_number
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.rounting_func = rounting_func
        self.rotate_func = rotate_func
        # self.align_weight = nn.Parameter(torch.randn(out_channels, in_channels, 1, 1),  # Shape for 1x1 convolution
        # requires_grad=True)
        # import pdb; pdb.set_trace()
        self.weight = nn.Parameter(

            torch.Tensor(
                kernel_number, 
                out_channels,
                in_channels // groups,
                kernel_size,
                kernel_size,
                
            )
        )

        
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
    


    def forward(self, x):
        # get alphas, angles
        # # [bs, Cin, h, w] --> [bs, n_theta], [bs, n_theta]
        # import pdb; pdb.set_trace()
        alphas, angles = self.rounting_func(x)       # shape:12 个批次的 1D 张量
        rotated_weight = self.rotate_func(self.weight, alphas, angles)#生成“旋转后的卷积核权重   [768, 64, 3, 3]
        #rotated_weight_flipped = torch.flip(rotated_weight, dims=[-1])#对卷积核翻转

        #import pdb; pdb.set_trace()
        bs, Cin, h, w = x.shape#torch.Size([12, 64, 13, 13])
        x = x.reshape(1, bs * Cin, h, w)  # [1, bs * Cin, h, w] bs=12,cin=64   [1, 768, 28, 28]
        #x = x.reshape(1, Cin, h, w)  # [1, bs * Cin, h, w] bs=12,cin=64
    
    
    
        # adaptive conv over images using group conv
        #out = F.conv2d(input=x, weight=rotated_weight, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation,groups=(self.groups * bs))
        out = F.conv2d(input=x, weight=rotated_weight, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation,groups=(self.groups * bs))#torch.Size([768, 64, 3, 3])
        #import pdb; pdb.set_trace()
        #out = F.conv2d(input=x, weight=rotated_weight, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation,groups=1)
        
        
        #conv_rotated = F.conv2d(input=x, weight=rotated_weight, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation,groups=(self.groups * bs))#torch.Size([1, 768, 32, 32])

        #conv_flipped = F.conv2d(input=x, weight=rotated_weight_flipped , bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation,groups=(self.groups * bs))#torch.Size([1, 768, 32, 32])n

        out = out.reshape(bs, self.out_channels, *out.shape[2:])
        #import pdb; pdb.set_trace()
        #out = out.reshape(bs, self.out_channels, *out.shape[2:])

        #conv_rotated = conv_rotated.reshape(bs, self.out_channels, *conv_rotated.shape[2:])#torch.Size([12, 64, 32, 32])
        #conv_flipped = conv_flipped.reshape(bs, self.out_channels, *conv_flipped.shape[2:])#torch.Size([12, 64, 32, 32])
        #fused_feature = self.gating_module(conv_rotated, conv_flipped) #使用 GatingModule 融合卷积特征
        # import pdb; pdb.set_trace()
        return out
        #return fused_feature

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_number={kernel_number}'
            ', kernel_size={kernel_size}, stride={stride}, bias={bias}')
        
        if self.padding != (0,) * len([self.padding]):
            s += ', padding={padding}'
        if self.dilation != (1,) * len([self.dilation]):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        return s.format(**self.__dict__)
    
    def initialize(self):
        weight_init(self)

class AdaptiveRotatedConv2d_attention(nn.Module):    
    def __init__(self, in_channels, out_channels, kernel_size,
        stride=1, padding=1, dilation=1, groups=1, bias=False,
        kernel_number=1, rounting_func=None, rotate_func=batch_rotate_multiweight):   #因为尺寸问题 需要添加二倍的初始化 
        super().__init__()
        self.kernel_number = kernel_number
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.attngnn = AttenMultiHead(64, 2)
        self.rounting_func = rounting_func
        self.rotate_func = rotate_func
        self.weight = nn.Parameter(
            torch.Tensor(
                kernel_number, 
                out_channels,
                in_channels // groups,
                kernel_size,
                kernel_size,
            )
        ) 
        self.conv_han1 = nn.Conv2d(64, 64, 4, stride=2, padding=1, bias=False)   #如果不跑attention需要注释掉下面四段
        self.conv_han2 = nn.Conv2d(64, 64, 4, stride=2, padding=1, bias=False)   #下采样,卷积操作中使用步幅（stride）时，可以跳过部分位置的计算，从而减少计算量。
        pool=(2,2)
        self.pool = nn.MaxPool2d(pool)
        
        
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # get alphas, angles
        # # [bs, Cin, h, w] --> [bs, n_theta], [bs, n_theta]
        # import pdb; pdb.set_trace()
        alphas, angles = self.rounting_func(x)       # shape:12 个批次的 1D 张量

        # rotate weight
        # # [Cout, Cin, k, k] --> [bs * Cout, Cin, k, k]
        # print(self.weight.shape)
        
        rotated_weight = self.rotate_func(self.weight, alphas, angles)#生成“旋转后的卷积核权重
        
        rotated_weight_fliped = torch.flip(rotated_weight, dims=[-1])#对生成“旋转后的卷积核权重翻转
        
        # combined_weight = torch.cat([rotated_weight, rotated_weight_fliped], dim=0)#合并
        

        # reshape images #12,64,32,32
    
        
        bs, Cin, h, w = x.shape
        x = x.reshape(1, bs * Cin, h, w)  # [1, bs * Cin, h, w]
        # x = torch.cat([x, x], dim=0)#合并
        # adaptive conv over images using group conv
        # out = F.conv2d(input=x, weight=combined_weight, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation,groups=(self.groups * bs))
        
        out_rotated = F.conv2d(input=x, weight=rotated_weight, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation,groups=(self.groups * bs))   #[1, 768, 28, 28])
        out_fliped = F.conv2d(input=x, weight=rotated_weight_fliped , bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation,groups=(self.groups * bs)) #([1, 768, 28, 28])


        out_rotated = out_rotated.reshape(bs, self.out_channels, *out_rotated.shape[2:])   #[12, 64, 44, 44]
        out_fliped = out_fliped.reshape(bs, self.out_channels, *out_fliped.shape[2:])   #[12, 64, 44, 44]

        

        # size_fliped = out_fliped.shape
        # hhh = self.conv_han1(out_rotated)       #[12, 64, 16, 16]
        
        out_rotated_pool_1 = self.conv_han1(out_rotated)   #[12, 64, 16, 16]
        out_fliped_pool_1 = self.conv_han2(out_fliped)  #[12, 64, 16, 16]

        # out_rotated_pool = self.pool(out_rotated_conv1)
        # out_fliped_pool=  self.pool(out_fiped_conv1)
        
        # size_rotate = out_rotated_pool.shape    #[12, 64, 16, 16]
    
        out_rotated_pool = out_rotated_pool_1.reshape(out_rotated_pool_1.shape[0], out_rotated_pool_1.shape[1], -1).permute(0, 2, 1) ## bs, n, k  [12, 256, 64]
        out_fliped_pool =  out_fliped_pool_1.reshape( out_fliped_pool_1 .shape[0], out_fliped_pool_1.shape[1], -1).permute(0, 2, 1)  #[12, 256, 64]

        out_attention = self.attngnn(out_fliped_pool,out_rotated_pool,out_rotated_pool,False) #[12, 256, 64]
        
        out_attention = out_attention.permute(0, 2, 1)           #[12, 64, 256]
        out_attention = out_attention.reshape(out_rotated_pool_1.shape[0], self.out_channels, *out_rotated_pool_1.shape[2:]) 

        out_attention_resized = F.interpolate(out_attention, size=out_fliped.shape[2:], mode='bilinear', align_corners=True)  #[12, 64, 16, 16]
        out_rotated = out_rotated + out_attention_resized    #  out_rotated=[12, 64, 32, 32]
        # import pdb; pdb.set_trace()

        # out = out_rotated.reshape(bs, self.out_channels, *tuple(out_rotated_pool_1.shape[2:]))
        return out_rotated
    
        #out = out.reshape(bs, self.out_channels, *out.shape[2:])        
        #return out           #值是卷积操作后的输出特征图，这个特征图表示输入数据经过 AdaptiveRotatedConv2d 卷积层后的结果

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_number={kernel_number}'
            ', kernel_size={kernel_size}, stride={stride}, bias={bias}')
            
        if self.padding != (0,) * len([self.padding]):
            s += ', padding={padding}'
        if self.dilation != (1,) * len([self.dilation]):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        return s.format(**self.__dict__)
    
    def initialize(self):
        weight_init(self)

class AdaptiveRotatedConv2d_sparse_attention(nn.Module):    
    def __init__(self, in_channels, out_channels, kernel_size,
        stride=1, padding=1, dilation=1, groups=1, bias=False,
        kernel_number=1, rounting_func=None, rotate_func=batch_rotate_multiweight):   #因为尺寸问题 需要添加二倍的初始化 
        super().__init__()
        self.kernel_number = kernel_number
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        # import pdb; pdb.set_trace()
        self.sparse_att = SparseAttention(4,1,local_attn_ctx=32,blocksize=8)
        self.rounting_func = rounting_func
        self.rotate_func = rotate_func
        self.weight = nn.Parameter(
            torch.Tensor(
                kernel_number, 
                out_channels,
                in_channels // groups,
                kernel_size,
                kernel_size,
            )
        ) 
        self.conv_han1 = nn.Conv2d(64, 64, 4, stride=2, padding=1, bias=False)   #如果不跑attention需要注释掉下面四段
        self.conv_han2 = nn.Conv2d(64, 64, 4, stride=2, padding=1, bias=False)   #下采样,卷积操作中使用步幅（stride）时，可以跳过部分位置的计算，从而减少计算量。
        pool=(2,2)
        self.pool = nn.MaxPool2d(pool)
        
        
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # get alphas, angles
        # # [bs, Cin, h, w] --> [bs, n_theta], [bs, n_theta]
        # import pdb; pdb.set_trace()
        alphas, angles = self.rounting_func(x)       # shape:12 个批次的 1D 张量

        # rotate weight
        # # [Cout, Cin, k, k] --> [bs * Cout, Cin, k, k]
        # print(self.weight.shape)
        
        rotated_weight = self.rotate_func(self.weight, alphas, angles)#生成“旋转后的卷积核权重
        
        rotated_weight_fliped = torch.flip(rotated_weight, dims=[-1])#对生成“旋转后的卷积核权重翻转
        
        # combined_weight = torch.cat([rotated_weight, rotated_weight_fliped], dim=0)#合并
        

        # reshape images #12,64,32,32
    
        
        bs, Cin, h, w = x.shape
        x = x.reshape(1, bs * Cin, h, w)  # [1, bs * Cin, h, w]
        # x = torch.cat([x, x], dim=0)#合并
        # adaptive conv over images using group conv
        # out = F.conv2d(input=x, weight=combined_weight, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation,groups=(self.groups * bs))
        
        out_rotated = F.conv2d(input=x, weight=rotated_weight, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation,groups=(self.groups * bs))   #[1, 768, 28, 28])
        out_fliped = F.conv2d(input=x, weight=rotated_weight_fliped , bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation,groups=(self.groups * bs)) #([1, 768, 28, 28])


        out_rotated = out_rotated.reshape(bs, self.out_channels, *out_rotated.shape[2:])   #[12, 64, 44, 44]
        out_fliped = out_fliped.reshape(bs, self.out_channels, *out_fliped.shape[2:])   #[12, 64, 44, 44]

        

        # size_fliped = out_fliped.shape
        # hhh = self.conv_han1(out_rotated)       #[12, 64, 16, 16]
        
        out_rotated_pool_1 = self.conv_han1(out_rotated)   #[12, 64, 16, 16]
        out_fliped_pool_1 = self.conv_han2(out_fliped)  #[12, 64, 16, 16]

        # out_rotated_pool = self.pool(out_rotated_conv1)
        # out_fliped_pool=  self.pool(out_fiped_conv1)
        
        # size_rotate = out_rotated_pool.shape    #[12, 64, 16, 16]
    
        out_rotated_pool = out_rotated_pool_1.reshape(out_rotated_pool_1.shape[0], out_rotated_pool_1.shape[1], -1).permute(0, 2, 1) ## bs, n, k  [12, 256, 64]
        out_fliped_pool =  out_fliped_pool_1.reshape( out_fliped_pool_1 .shape[0], out_fliped_pool_1.shape[1], -1).permute(0, 2, 1)  #[12, 256, 64]
        

        # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        out_attention = self.sparse_att(out_fliped_pool,out_rotated_pool,out_rotated_pool) #[12, 256, 64]
 
        out_attention = out_attention.permute(0, 2, 1)           #[12, 64, 256]
        out_attention = out_attention.reshape(out_rotated_pool_1.shape[0], self.out_channels, *out_rotated_pool_1.shape[2:]) 

        out_attention_resized = F.interpolate(out_attention, size=out_fliped.shape[2:], mode='bilinear', align_corners=True)  #[12, 64, 16, 16]
        out_rotated = out_rotated + out_attention_resized    #  out_rotated=[12, 64, 32, 32]

        # out = out_rotated.reshape(bs, self.out_channels, *tuple(out_rotated_pool_1.shape[2:]))
        return out_rotated
    
        #out = out.reshape(bs, self.out_channels, *out.shape[2:])        
        #return out           #值是卷积操作后的输出特征图，这个特征图表示输入数据经过 AdaptiveRotatedConv2d 卷积层后的结果

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_number={kernel_number}'
            ', kernel_size={kernel_size}, stride={stride}, bias={bias}')
            
        if self.padding != (0,) * len([self.padding]):
            s += ', padding={padding}'
        if self.dilation != (1,) * len([self.dilation]):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        return s.format(**self.__dict__)
    
    def initialize(self):
        weight_init(self)


