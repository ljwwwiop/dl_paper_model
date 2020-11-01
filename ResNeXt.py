import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

'''
    论文:CVPR2017
    实现ResNeXt(深度神经网络的聚合残差变换),这篇论文还是何恺明大牛团队的,基于ResNet结构提出的一种网络。
    提出理论：重复一个构建快来构建，该构建快聚合了一组具有相同拓扑结构的转换。这种策略称为“基数”
            即使高复杂度的条件下，增加基数也能提高分类精度。另外，增加容量，增加基数比增加深或宽更有效。
    ResNeXt采用split-transform-merge的思想,但是沿用VGG/ResNet的构造重复的卷积层的策略，使网络在具有性能的基础上更加优雅简洁。
    右图是ResNeXt的一个32x4d的基本结构，32指的是cardinality是32，即利用1x1卷积降维，并分成32条支路；
    4d指的是每个支路中transform的3x3卷积的滤波器数量为4。
    
'''

class BN_Conv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,dilation = 1,groups=1,bias=False):
        super(BN_Conv2d, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,bias),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return F.relu(self.seq(x))

# 关于block 结构，实现的时候需要主要 输入维度，cardinality(基数)，transform滤波器的数量
# 采用的分组卷积
# 第一个conv 是 256 1x1 128
class ResNeXt_Block(nn.Module):
    '''
    ResNeXt block with group convolutions
    '''
    def __init__(self,in_chnls,cardinality,group_depth,stride):
        super(ResNeXt_Block, self).__init__()
        self.group_chnls = cardinality * group_depth
        self.conv1 = BN_Conv2d(in_chnls,self.group_chnls,1,stride=1,padding=0)
        self.conv2 = BN_Conv2d(self.group_chnls,self.group_chnls,3,stride=stride,padding=1,groups=cardinality)
        self.conv3 = nn.Conv2d(self.group_chnls,self.group_chnls*2,1,stride=1,padding=0)
        self.bn = nn.BatchNorm2d(self.group_chnls*2)
        self.short_cut = nn.Sequential(
            nn.Conv2d(in_chnls,self.group_chnls*2,1,stride,0,bias=False),
            nn.BatchNorm2d(self.group_chnls*2)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.bn(out)
        out += self.short_cut(x)
        return F.relu(out)

class ResNeXt(nn.Module):
    '''
    ResNeXt builder
    '''
    def __init__(self,layers,cardinality,group_depth,num_classes):
        super(ResNeXt, self).__init__()
        self.channels = 64
        self.cardinality = cardinality
        self.conv1 = BN_Conv2d(3,self.channels,7,stride=2,padding=3)
        d1 = group_depth
        self.conv2 = self.__make_layers(d1,layers[0],stride=1)
        d2 = d1*2
        self.conv3 = self.__make_layers(d2,layers[1],stride=2)
        d3 = d2*2
        self.conv4 = self.__make_layers(d3,layers[2],stride=2)
        d4 = d3*2
        self.conv5 = self.__make_layers(d4,layers[3],stride=2)
        self.fc = nn.Linear(self.channels,num_classes) # 224x224 input size 有限制

    def __make_layers(self,d,blocks,stride):
        strides = [stride] + [1]*(blocks-1)
        layers = []
        for stride in strides:
            layers.append(ResNeXt_Block(self.channels,self.cardinality,d,stride))
            self.channels = self.cardinality*d*2
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = F.max_pool2d(out,3,2,1)
        out = self.conv5(self.conv4(self.conv3(self.conv2(out))))
        out = F.avg_pool2d(out,7)
        out = out.view(out.size(0),-1)
        out = F.softmax(self.fc(out),dim=1)
        return out

def resNeXt50_32x4d(num_classes=1000):
    return ResNeXt([3, 4, 6, 3], 32, 4, num_classes)


def resNeXt101_32x4d(num_classes=1000):
    return ResNeXt([3, 4, 23, 3], 32, 4, num_classes)


def resNeXt101_64x4d(num_classes=1000):
    return ResNeXt([3, 4, 23, 3], 64, 4, num_classes)

if __name__ == '__main__':
    net = resNeXt50_32x4d(10)
    summary(net,(3,224,224))



