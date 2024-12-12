import math
import einops
import torch
import torch.nn as nn
from .weight_init import trunc_normal_


class LayerNormProxy(nn.Module):
    # copy from https://github.com/LeapLabTHU/DAT/blob/main/models/dat_blocks.py
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')
    
class RountingFunction(nn.Module):

    def __init__(self, in_channels, kernel_number, dropout_rate=0.2, proportion=40.0):
        super().__init__()
        self.kernel_number = kernel_number
        # self.dwc = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,             
        #                      groups=in_channels, bias=False)
        self.dwc = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1, groups=1)

        self.norm = LayerNormProxy(in_channels)
        self.relu = nn.ReLU(inplace=True)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc_alpha = nn.Linear(in_channels, kernel_number, bias=True)

        self.dropout2= nn.Dropout(dropout_rate)
        self.fc_theta = nn.Linear(in_channels, kernel_number, bias=False)

        self.act_func = nn.Softsign()
        self.proportion = proportion / 180.0 * math.pi
        
        # init weights
        trunc_normal_(self.dwc.weight, std=.02)
        trunc_normal_(self.fc_alpha.weight, std=.02)
        trunc_normal_(self.fc_theta.weight, std=.02)

    def forward(self, x):

        x = self.dwc(x)
        x = self.norm(x)
        x = self.relu(x)

        x = self.avg_pool(x).squeeze(dim=-1).squeeze(dim=-1)  # avg_x.shape = [batch_size, Cin]

        alphas = self.dropout1(x)
        alphas = self.fc_alpha(alphas)
        alphas = torch.sigmoid(alphas)

        angles = self.dropout2(x)
        angles = self.fc_theta(angles)
        angles = self.act_func(angles)
        angles = angles * self.proportion

        return alphas, angles

class RountingFunction_flip(nn.Module):

    def __init__(self, in_channels, kernel_number, dropout_rate=0.2, proportion=40.0):
        super().__init__()
        self.kernel_number = kernel_number
        # self.dwc = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,             
        #                      groups=in_channels, bias=False)
        self.dwc = nn.Conv2d(in_channels=64, out_channels=in_channels, kernel_size=3, padding=1, groups=1)#原始卷积核

        self.norm = LayerNormProxy(in_channels)
        self.relu = nn.ReLU(inplace=True)
        

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc_alpha = nn.Linear(in_channels, kernel_number, bias=True)

        self.dropout2= nn.Dropout(dropout_rate)
        self.fc_theta = nn.Linear(in_channels, kernel_number, bias=False)
        
        
        self.dropout3= nn.Dropout(dropout_rate)#alphas_flip
        self.fc_alpha_flip = nn.Linear(in_channels, kernel_number, bias=True)#alphas_flip

        self.act_func = nn.Softsign()
        self.proportion = proportion / 180.0 * math.pi
        
        # init weights
        trunc_normal_(self.dwc.weight, std=.02)
        trunc_normal_(self.fc_alpha.weight, std=.02)
        trunc_normal_(self.fc_theta.weight, std=.02)

    def forward(self, x):

        x = self.dwc(x)
        x = self.norm(x)
        x = self.relu(x)
        #x_flip = self.norm_flip(x)

        x = self.avg_pool(x).squeeze(dim=-1).squeeze(dim=-1)  # avg_x.shape = [batch_size, Cin]
        
        alphas = self.dropout1(x)
        alphas = self.fc_alpha(alphas)
        alphas = torch.sigmoid(alphas)
    


        alphas_flip = self.dropout3(x)
        alphas_flip = self.fc_alpha_flip(alphas_flip)
        alphas_flip = torch.sigmoid(alphas_flip)
        
        

        angles = self.dropout2(x)
        angles = self.fc_theta(angles)
        angles = self.act_func(angles)
        angles = angles * self.proportion

        return alphas, angles, alphas_flip
    
    def extra_repr(self):
        s = (f'kernel_number={self.kernel_number}')
        return s.format(**self.__dict__)


class RountingFunction_flip_conv(nn.Module):

    def __init__(self, in_channels, kernel_number, dropout_rate=0.2, proportion=40.0):
        super().__init__()
        self.kernel_number = kernel_number
        # self.dwc = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,             
        #                      groups=in_channels, bias=False)
        self.dwc = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1, groups=1)#原始卷积核

        self.norm = LayerNormProxy(in_channels)
        self.relu = nn.ReLU(inplace=True)
        

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.dwc1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1, groups=1)#原始卷积核

        self.norm1 = LayerNormProxy(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        

        self.avg_pool1 = nn.AdaptiveAvgPool2d((1, 1))


        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc_alpha = nn.Linear(in_channels, kernel_number, bias=True)

        self.dropout2= nn.Dropout(dropout_rate)
        self.fc_theta = nn.Linear(in_channels, kernel_number, bias=False)
        
        
        self.dropout3= nn.Dropout(dropout_rate)#alphas_flip
        self.fc_alpha_flip = nn.Linear(in_channels, kernel_number, bias=True)#alphas_flip

        self.act_func = nn.Softsign()
        self.proportion = proportion / 180.0 * math.pi
        
        # init weights
        trunc_normal_(self.dwc.weight, std=.02)
        trunc_normal_(self.fc_alpha.weight, std=.02)
        trunc_normal_(self.fc_theta.weight, std=.02)

    def forward(self, x):
        x1 =x
        
        x = self.dwc(x)
        x = self.norm(x)
        x = self.relu(x)
        
        
        x1 = self.dwc1(x1)
        x1 = self.norm1(x1)
        x1 = self.relu1(x1)
        x1 = self.avg_pool1(x1).squeeze(dim=-1).squeeze(dim=-1)  # avg_x.shape = [batch_size, Cin]
        #x_flip = self.norm_flip(x)

        x = self.avg_pool(x).squeeze(dim=-1).squeeze(dim=-1)  # avg_x.shape = [batch_size, Cin]
        
        alphas = self.dropout1(x)
        alphas = self.fc_alpha(alphas)
        alphas = torch.sigmoid(alphas)
    


        alphas_flip = self.dropout3(x1)
        alphas_flip = self.fc_alpha_flip(alphas_flip)
        alphas_flip = torch.sigmoid(alphas_flip)

        angles = self.dropout2(x)
        angles = self.fc_theta(angles)
        angles = self.act_func(angles)
        angles = angles * self.proportion

        return alphas, angles, alphas_flip
    
    def extra_repr(self):
        s = (f'kernel_number={self.kernel_number}')
        return s.format(**self.__dict__)


class RountingFunction_flip_conv_angle(nn.Module):

    def __init__(self, in_channels, kernel_number, dropout_rate=0.2, proportion=40.0):
        super().__init__()
        self.kernel_number = kernel_number
        # self.dwc = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,             
        #                      groups=in_channels, bias=False)
        self.dwc = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1, groups=1)

        self.norm = LayerNormProxy(in_channels)
        self.relu = nn.ReLU(inplace=True)
        

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.dwc1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1, groups=1)

        self.norm1 = LayerNormProxy(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        

        self.avg_pool1 = nn.AdaptiveAvgPool2d((1, 1))


        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc_alpha = nn.Linear(in_channels, kernel_number, bias=True)

        self.dropout2= nn.Dropout(dropout_rate)
        self.fc_theta = nn.Linear(in_channels, kernel_number, bias=False)
        
        
        self.dropout3= nn.Dropout(dropout_rate)#alphas_flip
        self.fc_alpha_flip = nn.Linear(in_channels, kernel_number, bias=True)#alphas_flip

        self.act_func = nn.Softsign()
        self.proportion = proportion / 180.0 * math.pi
        
        # init weights
        trunc_normal_(self.dwc.weight, std=.02)
        trunc_normal_(self.fc_alpha.weight, std=.02)
        trunc_normal_(self.fc_theta.weight, std=.02)

    def forward(self, x):
        x1 =x
        
        x = self.dwc(x)
        x = self.norm(x)
        x = self.relu(x)
        
        
        x1 = self.dwc1(x1)
        x1 = self.norm1(x1)
        x1 = self.relu1(x1)
        x1 = self.avg_pool1(x1).squeeze(dim=-1).squeeze(dim=-1)  # avg_x.shape = [batch_size, Cin]
        #x_flip = self.norm_flip(x)

        x = self.avg_pool(x).squeeze(dim=-1).squeeze(dim=-1)  # avg_x.shape = [batch_size, Cin]
        
        alphas = self.dropout1(x)
        alphas = self.fc_alpha(alphas)
        alphas = torch.sigmoid(alphas)
    


        alphas_flip = self.dropout3(x1)
        alphas_flip = self.fc_alpha_flip(alphas_flip)
        alphas_flip = torch.sigmoid(alphas_flip)

        angles_flip = self.dropout2(x1)
        angles_flip = self.fc_theta(angles_flip)
        angles_flip = self.act_func(angles_flip)
        angles_flip = angles_flip * self.proportion



        angles = self.dropout2(x)
        angles = self.fc_theta(angles)
        angles = self.act_func(angles)
        angles = angles * self.proportion

        return alphas, angles, alphas_flip, angles_flip
    
    def extra_repr(self):
        s = (f'kernel_number={self.kernel_number}')
        return s.format(**self.__dict__)

