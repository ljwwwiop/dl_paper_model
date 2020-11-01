import torch.nn as nn
import torch.nn.functional as F
import math

'''
    论文：FPN
    关于实现一下FPN网络（金字塔特征网络）
    backbone - ResNet网络
    FPN特点：网络模型主要是三个部分组成（自底向上 + 自顶向下 + 中间部分）
    不同尺寸的特征图进行融合后，可以增强学习效率，并且减少了计算量。
'''

__all__ = ['FPN']

# 残差块
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,in_planes,planes,stride = 1,downsample = None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes,planes,kernel_size=1,bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,planes,kernel_size=3,stride=stride,padding=1,bias = False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes,self.expansion * planes ,kernel_size=1,bias = False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.relu = nn.ReLU(inplace= True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class FPN(nn.Module):
    def __init__(self,block,layers):
        super(FPN, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2,padding=1)
        # 自底向上 layers 除去c1 层，总共还有4层
        self.layer1 = self._make_layer(block,64,layers[0])
        self.layer2 = self._make_layer(block,128,layers[1],stride=2)
        self.layer3 = self._make_layer(block,256,layers[2],stride=2)
        self.layer4 = self._make_layer(block,512,layers[3],stride=2)

        # 顶层 1x1x256 Reduce channels -> 256  -> p5
        self.toplayer = nn.Conv2d(2048,256,kernel_size=1,stride=1,padding=0)
        # 三层3x3卷积特征提取层，主要是消除多层pooling的叠化效果
        self.smooth1 = nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1)
        self.smooth2 = nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1)
        self.smooth3 = nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1)
        # 三层连接层 属于中间部分
        self.latlayer1 = nn.Conv2d(1024,256,kernel_size=1,stride=1,padding=0)
        self.latlayer2 = nn.Conv2d(512,256,kernel_size=1,stride=1,padding=0)
        self.latlayer3 = nn.Conv2d(256,256,kernel_size=1,stride=1,padding=0)

        for m in self.modules():
            # 初始化参数
            if isinstance(m,nn.Conv2d):
                n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0,math.sqrt(2./n))
            elif isinstance(m,nn.BatchNorm2d):
                # 初始化
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self,block,planes,blocks,stride = 1):
        downsample = None

        if stride!=1 or self.inplanes != block.expansion * planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,block.expansion*planes,kernel_size=1,stride = stride,bias=False),
                nn.BatchNorm2d(block.expansion*planes)
            )
        layers = []
        layers.append(block(self.inplanes,planes,stride,downsample))
        self.inplanes = planes * block.expansion
        for i in range(1,blocks):
            layers.append(block(self.inplanes,planes))
        return nn.Sequential(*layers)

    def _upsample_add(self,x,y):
        # 上采样
        _,_,H,W = y.size()
        return F.upsample(x,size=(H,W),mode='bilinear') + y

    def forward(self, x):
        # 自底向上
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        c1 = self.maxpool(x)

        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        # 自顶向下
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5,self.latlayer1(c4))
        p3 = self._upsample_add(p4,self.latlayer2(c3))
        p2 = self._upsample_add(p3,self.latlayer3(c2))
        # smooth层 提取特征
        p4 = self.smooth1(p4) # 上
        p3 = self.smooth2(p3) # 中
        p2 = self.smooth3(p2) # 下

        return p2,p3,p4,p5

def FPN101():
    return FPN(Bottleneck,[2,2,2,2])

