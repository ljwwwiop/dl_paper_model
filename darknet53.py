import torch
import torch.nn as nn

'''
    论文：Yolo v3
    实现关于Darknet-53网络
    特点： 最后fc前面没有pooling，也是引用残差结构，整个网络结构只有两种卷积核
'''

def Conv3x3BNReLU(in_channels,out_channels,stride=1):
    # 残差块中的3x3
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def Conv1x1BNReLU(in_channels,out_channels):
    # 残差块中的1x1
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class Residual(nn.Module):
    # 残差块
    def __init__(self,nchannels):
        super(Residual, self).__init__()
        mid_channels = nchannels//2
        self.conv1x1 = Conv1x1BNReLU(nchannels,mid_channels)
        self.conv3x3 = Conv3x3BNReLU(mid_channels,nchannels)

    def forward(self, x):
        out = self.conv3x3(self.conv1x1(x))
        return out + x

class Darknet53(nn.Module):
    def __init__(self,num_classes = 10):
        super(Darknet53, self).__init__()
        # [32 3x3 256x256]
        self.first_conv = Conv3x3BNReLU(in_channels=3,out_channels=32)
        # print(self.first_conv)
        # 一个block包括了[卷积 64 3x3 /2 128x128 整个残差块x1]
        self.block1 = self._make_layers(in_channels=32,out_channels=64,block_num=1)
        # print(self.block1)
        self.block2 = self._make_layers(in_channels=64,out_channels=128,block_num=2)
        self.block3 = self._make_layers(in_channels=128,out_channels=256,block_num=8)
        self.block4 = self._make_layers(in_channels=256,out_channels=512,block_num=8)
        self.block5 = self._make_layers(in_channels=512,out_channels=1024,block_num=4)

        self.avg_pool = nn.AvgPool2d(kernel_size=8,stride=1)
        self.linear = nn.Linear(in_features=1024*8*8,out_features=num_classes)
        self.softmax = nn.Softmax(dim=1)

    def _make_layers(self,in_channels,out_channels,block_num):
        # print("显示当前block区的每个层内容")
        _layers = []
        _layers.append(Conv3x3BNReLU(in_channels,out_channels,stride=2))
        for _ in range(block_num):
            _layers.append(Residual(nchannels=out_channels))
        return nn.Sequential(*_layers)

    def forward(self, x):
        x = self.first_conv(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        # print(x.shape,x.view(x.size(0),-1))
        x = x.view(x.size(0),-1)
        x = self.linear(x)
        out = self.softmax(x)
        return out

if __name__ == '__main__':
    model = Darknet53()
    print(model)
    input = torch.randn(1,3,256,256)
    out = model(input)
    print("out shape")
    print(out.shape)
    print(out)



