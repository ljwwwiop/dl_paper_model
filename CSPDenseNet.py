import torch.nn as nn
import torch
import torch.nn.functional as F

'''
    论文：CSPNet
    实现CSPDenseNet网络
    CSP是在DenseNet基础上改进的一种网络
    特点：减少大量重复的梯度计算，并且提高网络性能。
    也是一种更好的不改变网络基本原理，从而提高网络能力的结构。
    
'''
# 作者将激活函数采用LeakyReLU,但在一些目标检测框架中会用Mish 作为激活函数，以更高的计算代价获取网络更好的收敛性
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

class BN_Conv2d(nn.Module):
    '''
    BN_CONV_RELU
    '''
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,dilation = 1,groups =1,bias = False):
        super(BN_Conv2d, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,bias),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return F.relu(self.seq(x))

# Partial Dense Block
class DenseBlock(nn.Module):

    def __init__(self,input_channels,num_layers,growth_rate):
        super(DenseBlock, self).__init__()
        self.num_layers = num_layers
        self.k0 = input_channels
        self.k = growth_rate
        self.layers = self.__make_layers()

    def __make_layers(self):
        layer_list = []
        for i in range(self.num_layers):
            layer_list.append(nn.Sequential(
                BN_Conv2d(self.k0 + i*self.k,4*self.k,1,1,0),
                BN_Conv2d(4*self.k,self.k,3,1,1)
            ))
        return layer_list

    def forward(self, x):
        feature = self.layers[0](x)
        out = torch.cat((x,feature),1)
        for i in range(1,len(self.layers)):
            feature = self.layers[i](out)
            out = torch.cat((feature,out),1)
        return out

# Partial Dense Block
# 采用Fusion Last的 Partial transition layer
class CSP_DenseBlock(nn.Module):

    def __init__(self,in_channels,num_layers,k,part_ratio = 0.5):
        super(CSP_DenseBlock, self).__init__()
        self.part1_chnls = int(in_channels*part_ratio)
        self.part2_chnls = in_channels - self.part1_chnls
        self.dense = DenseBlock(self.part2_chnls,num_layers,k)
        # trans_chnls = self.part2_chnls + k * num_layers
        # self.transtion = BN_Conv2d(trans_chnls, trans_chnls, 1, 1, 0)

    def forward(self, x):
        part1 = x[:,:self.part1_chnls,:,:]
        part2 = x[:,self.part1_chnls:,:,:]
        part2 = self.dense(part2)
        out = torch.cat((part1,part2),1)
        return out

class DenseNet(nn.Module):
    def __init__(self,layers,k,theta,num_classes,part_ratio = 0):
        super(DenseNet, self).__init__()
        # params
        self.layers = layers
        self.k = k
        self.theta = theta
        self.Block = DenseBlock if part_ratio ==0 else CSP_DenseBlock # 通过part_ratio 控制block
        # layers
        self.conv = BN_Conv2d(3,2*k,7,2,3)
        self.blocks,patches = self.__make_blocks(2*k)
        self.fc = nn.Linear(patches,num_classes)

    def __make_transition(self,in_chls):
        out_chls = int(self.theta * in_chls)
        return nn.Sequential(
            BN_Conv2d(in_chls,out_chls,1,1,0),
            nn.AvgPool2d(2)
        ),out_chls

    def __make_blocks(self,k0):
        '''
        make block- transition structures
        :param k0:
        :return:
        '''
        layers_list = []
        patches = 0
        for i in range(len(self.layers)):
            layers_list.append(self.Block(k0,self.layers[i],self.k))
            patches = k0 + self.layers[i] * self.k
            if i != len(self.layers) -1:
                transition ,k0 = self.__make_transition(patches)
                layers_list.append(transition)
        return nn.Sequential(*layers_list),patches

    def forward(self, x):
        out = self.conv(x)
        out = F.max_pool2d(out,3,2,1)
        print(" - " ,out.shape)
        out = self.blocks(out)
        out = F.avg_pool2d(out,7)
        print("二 ",out.shape)
        out = out.view(out.size(0),-1)
        out = F.softmax(self.fc(out),dim=1)
        return out

def csp_densenet_121(num_classes = 1000):
    return DenseNet([6,12,24,16],k=32,theta=0.5,num_classes=num_classes,part_ratio=0.5)

def csp_densenet_169(num_classes=1000):
    return DenseNet([6, 12, 32, 32], k=32, theta=0.5, num_classes=num_classes, part_ratio=0.5)

def csp_densenet_201(num_classes=1000):
    return DenseNet([6, 12, 48, 32], k=32, theta=0.5, num_classes=num_classes, part_ratio=0.5)

def csp_densenet_264(num_classes=1000):
    return DenseNet([6, 12, 64, 48], k=32, theta=0.5, num_classes=num_classes, part_ratio=0.5)

if __name__ == "__main__":
    model = csp_densenet_121(10)
    print(model)
    input = torch.randn(1,3,256,256)
    out = model(input)
    print(out)
    print(out.shape)

