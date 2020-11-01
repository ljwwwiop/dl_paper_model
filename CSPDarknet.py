import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

'''
    
    实现YOLOv4特征提取网络 - CSPDarkNet结构
    特点：Mish激活函数
    问题：CSP结构，特征输入后，通过一个比例将其分为两个部分，然后再分别输入block结构，以及后面的Partial transition处理。
    
'''

# 激活函数Mish
class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()
    def forward(self, x):
        return x *torch.tanh(F.softplus(x))

class BN_Conv_Mish(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,dilation=1,groups=1,bias=False):
        super(BN_Conv_Mish, self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,bias)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.bn(self.conv(x))
        return Mish()(out)

class ResidualBlock(nn.Module):
    '''
    basic residual block for CSP-Darknet
    '''
    def __init__(self,chnls,inner_chnls = None):
        super(ResidualBlock, self).__init__()
        if inner_chnls is None:
            inner_chnls = chnls
        self.conv1 = BN_Conv_Mish(chnls,inner_chnls,1,1,0)
        self.conv2 = BN_Conv_Mish(inner_chnls,chnls,3,1,1,bias=False)
        self.bn = nn.BatchNorm2d(chnls)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn(out) + x
        return Mish()(out)

class CSPFirst(nn.Module):
    '''
    First CSP Stage
    '''
    def __init__(self,in_chnls,out_chnls):
        super(CSPFirst, self).__init__()
        self.downsample = BN_Conv_Mish(in_chnls,out_chnls,3,2,1)
        self.trans_0 = BN_Conv_Mish(out_chnls,out_chnls,1,1,0)
        self.trans_1 = BN_Conv_Mish(out_chnls,out_chnls,1,1,0)
        self.blocks = ResidualBlock(out_chnls,out_chnls//2)
        self.trans_cat = BN_Conv_Mish(2*out_chnls,out_chnls,1,1,0)

    def forward(self, x):
        x = self.downsample(x)
        out_0 = self.trans_0(x)
        out_1 = self.trans_1(x)
        out = self.blocks(out_1)
        out = torch.cat((out_0,out_1),1)
        out = self.trans_cat(out)
        return out


class CSPStem(nn.Module):
    """
    CSP structures including downsampling
    """
    def __init__(self, in_chnls, out_chnls, num_block):
        super(CSPStem, self).__init__()
        self.dsample = BN_Conv_Mish(in_chnls, out_chnls, 3, 2, 1)
        self.trans_0 = BN_Conv_Mish(out_chnls, out_chnls // 2, 1, 1, 0)
        self.trans_1 = BN_Conv_Mish(out_chnls, out_chnls // 2, 1, 1, 0)
        # print("self.trans_0",self.trans_0)
        # print("self.trans_1",self.trans_1)
        self.blocks = nn.Sequential(*[ResidualBlock(out_chnls // 2) for _ in range(num_block)])
        self.trans_cat = BN_Conv_Mish(out_chnls, out_chnls, 1, 1, 0)

    def forward(self, x):
        x = self.dsample(x)
        out_0 = self.trans_0(x)
        out_1 = self.trans_1(x)
        out_1 = self.blocks(out_1)
        # print("self.trans_0",self.trans_0)
        # print("self.trans_1",self.trans_1)
        # print("self.blocks",self.blocks)
        print("out_0",out_0.shape)
        print("out_1", out_1.shape)
        out = torch.cat((out_0, out_1), 1)
        print("out",out.shape)
        out = self.trans_cat(out)
        return out

class CSP_DarkNet(nn.Module):
    '''
    CSP-DarkNet
    '''
    def __init__(self,num_blocks,num_classes = 1000):
        super(CSP_DarkNet, self).__init__()
        chnls = [64, 128, 256, 512, 1024]
        self.conv0 = BN_Conv_Mish(3,32,3,1,1)
        self.neck = CSPFirst(32,chnls[0])
        # print(self.neck)
        self.body = nn.Sequential(
            *[CSPStem(chnls[i], chnls[i+1], num_blocks[i]) for i in range(4)]
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(chnls[4],num_classes)

    def forward(self, x):
        out = self.conv0(x)
        out = self.neck(out)
        out = self.body(out)
        out = self.global_pool(out)
        out = out.view(out.size(0),-1)
        out = self.fc(out)
        return F.softmax(out,dim=1)

def csp_darknet_53(num_classes=1000):
    return CSP_DarkNet([2, 8, 8, 4], num_classes)

if __name__ == '__main__':
    net = csp_darknet_53()
    summary(net, (3, 256, 256))






