import torch
import torch.nn as nn
import torch.nn.functional as F

'''
    论文：CSPNet
    类似CSPDenseNet差不多的原理，在Resnet基础上搭配上了CSP
    CSP主要是将需要重复计算的梯度信息消除掉
    将base layer 一分为二 两块，一步分进入 resnet layer 剩下一步分 Partial Transition layer
    最后一步分才进行融合
'''
class BN_Conv2d_Leaky(nn.Module):
    """
    BN_CONV_LeakyRELU
    """
    def __init__(self, in_channels: object, out_channels: object, kernel_size: object, stride: object, padding: object,
                 dilation=1, groups=1, bias=False) -> object:
        super(BN_Conv2d_Leaky, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, groups=groups, bias=bias),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return F.leaky_relu(self.seq(x))

class ResidualBlock(nn.Module):
    """
    Residual block for CSP-ResNeXt
    """
    def __init__(self, in_channels, cardinality, group_width, stride=1):
        super(ResidualBlock, self).__init__()
        self.out_channels = cardinality * group_width
        self.conv1 = BN_Conv2d_Leaky(in_channels, self.out_channels, 1, 1, 0)
        self.conv2 = BN_Conv2d_Leaky(self.out_channels, self.out_channels, 3, stride, 1, groups=cardinality)
        self.conv3 = nn.Conv2d(self.out_channels, self.out_channels, 1, 1, 0)
        self.bn = nn.BatchNorm2d(self.out_channels)

        # make shortcut
        layers = []
        if in_channels != self.out_channels:
            layers.append(nn.Conv2d(in_channels, self.out_channels, 1, 1, 0))
            layers.append(nn.BatchNorm2d(self.out_channels))
        if stride != 1:
            layers.append(nn.AvgPool2d(stride))
        self.shortcut = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv3(self.conv2(self.conv1(x)))
        out = self.bn(out)
        out += self.shortcut(x)
        return F.leaky_relu(out)

class Stem(nn.Module):
    def __init__(self, in_channels, num_blocks, cardinality, group_with, stride=2):
        super(Stem, self).__init__()
        self.c0 = in_channels // 2
        self.c1 = in_channels - in_channels // 2
        self.hidden_channels = cardinality * group_with
        self.out_channels = self.hidden_channels * 2
        self.trans_part0 = nn.Sequential(BN_Conv2d_Leaky(self.c0, self.hidden_channels, 1, 1, 0), nn.AvgPool2d(stride))
        self.block = self.__make_block(num_blocks, self.c1, cardinality, group_with, stride)
        self.trans_part1 = BN_Conv2d_Leaky(self.hidden_channels, self.hidden_channels, 1, 1, 0)
        self.trans = BN_Conv2d_Leaky(self.out_channels, self.out_channels, 1, 1, 0)

    def __make_block(self, num_blocks, in_channels, cardinality, group_with, stride):
        strides = [stride] + [1] * (num_blocks-1)
        channels = [in_channels] + [self.hidden_channels] * (num_blocks-1)
        return nn.Sequential(*[ResidualBlock(c, cardinality, group_with, s)
                               for c, s in zip(channels, strides)])

    def forward(self, x):
        print("x ",x.shape)
        x0 = x[:, :self.c0, :, :]
        x1 = x[:, :self.c0, :, :]
        out0 = self.trans_part0(x0)
        print("out0 ",out0.shape)
        # print("out0",out0)
        t = self.block(x1)
        print("t",t.shape)
        out1 = self.trans_part1(t)
        print("out1 ",out1.shape)
        # print("out1",out1)
        out = torch.cat((out0, out1), 1)
        print("out ",out.shape)
        # print("out ",out)
        return self.trans(out)

class CSP_ResNeXt(nn.Module):
    def __init__(self, num_blocks, cadinality, group_width, num_classes):
        super(CSP_ResNeXt, self).__init__()
        self.conv0 = BN_Conv2d_Leaky(3, 64, 7, 2, 3)
        self.pool1 = nn.MaxPool2d(3, 2, 1)
        self.conv1 = BN_Conv2d_Leaky(64, 128, 1, 1, 0)
        self.stem0 = Stem(cadinality*group_width*2, num_blocks[0], cadinality, group_width, stride=1)
        self.stem1 = Stem(cadinality*group_width*4, num_blocks[1], cadinality, group_width*2)
        self.stem2 = Stem(cadinality*group_width*8, num_blocks[2], cadinality, group_width*4)
        self.stem3 = Stem(cadinality*group_width*16, num_blocks[3], cadinality, group_width*8)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(cadinality*group_width*16, num_classes)

    def forward(self, x):
        out = self.conv0(x)
        out = self.pool1(out)
        out = self.conv1(out)
        out = self.stem0(out)
        out = self.stem1(out)
        out = self.stem2(out)
        out = self.stem3(out)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return F.softmax(out,dim=1)

def csp_resnext_50_32x4d(num_classes=10):
    return CSP_ResNeXt([3, 4, 6, 3], 32, 4, num_classes)

if __name__ == '__main__':
    net = csp_resnext_50_32x4d()
    print(net)
    input = torch.randn(1,3,256,256)
    out = net(input)
    print(out.shape)
    print(out)



