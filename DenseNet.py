import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

'''
    论文：DenseNet CVPR2018
    实现DenseNet网络,主要是密集网络
    xl = H([x0,x1,x2....xl-1])
    特点：将之前每一层的特征输出作为该层的输入，而且之前层不是通过加法，而是通过叠加的方式组合的。
    滤波器数量 称为 Growth rate，表示为K，每个block中第I层的输入为k_0 + (I-1)k,k_0是初始输入的维度
    DenseNet优点明显：进一步缓解梯度消失的问题，强化了特征传递，促进了不同特征的融合，k设置较小可以实现很好的性能，减少参数量
                缺点：计算量蛮大
    Bottleneck -> 是Dense Block采用的结构
    卷积层：BN-ReLU - Conv
    可以改进地方：pool可以引入SPP,FPN试一试
'''
class BN_Conv2d(nn.Module):
    '''
    BN_CONV_RELU
    '''
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,dilation = 1,groups = 1,bias = False):
        super(BN_Conv2d, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,bias),
            nn.BatchNorm2d(out_channels)
            # 也可以直接使用nn.ReLU()
        )

    def forward(self, x):
        return F.relu(self.seq(x))

class DenseBlock(nn.Module):
    '''
    DenseBlock 通过grow rate控制，卷积层数量L，输入特征层三个参数控制
    '''
    def __init__(self,input_channels,num_layers,growth_rate):
        super(DenseBlock, self).__init__()
        self.num_layers = num_layers
        self.k0 = input_channels
        self.k = growth_rate
        self.layers = self.__make_layers()
        print()

    def __make_layers(self):
        # 构建Dense Block
        layer_list = []
        for i in range(self.num_layers):
            layer_list.append(nn.Sequential(
                BN_Conv2d(self.k0 + i *self.k,4 *self.k,1,1,0),
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

# 网络主干
class DenseNet(nn.Module):

    def __init__(self,layers,k,theta,num_classes):
        super(DenseNet, self).__init__()
        # params
        self.layers = layers
        self.k = k
        self.theta = theta
        # layers
        self.conv = BN_Conv2d(3,2*k,7,2,3)
        self.blocks,patches = self.__make_blocks(2*k)
        self.fc = nn.Linear(patches,num_classes)

    def __make_transition(self,in_chls):
        # 连接两个dense blocks的
        out_chls = int(self.theta * in_chls)
        return nn.Sequential(
            BN_Conv2d(in_chls,out_chls,1,1,0),
            nn.AvgPool2d(2)
        ),out_chls

    def __make_blocks(self,k0):
        '''
        make block transition
        :param k0:
        :return:
        '''
        layers_list = []
        patches = 0
        for i in range(len(self.layers)):
            layers_list.append(DenseBlock(k0,self.layers[i],self.k))
            patches = k0 + self.layers[i]*self.k  # output feature patches from Dense Block
            if i != len(self.layers) - 1:
                transition , k0 = self.__make_transition(patches)
                layers_list.append(transition)
        return nn.Sequential(*layers_list),patches

    def forward(self, x):
        out = self.conv(x)
        out = F.max_pool2d(out,3,2,1)

        out = self.blocks(out)
        out = F.avg_pool2d(out,7)
        out = out.view(out.size(0),-1)
        out = F.softmax(self.fc(out),dim=1)
        return out

def densenet_121(num_classes=1000):
    # k - growth rate 滤波个数 theta 控制输出层[theta * input_channels]
    # 将k设置得较小即可实现很好的性能，显著减少了网络的参数量
    return DenseNet([6, 12, 24, 16], k=32, theta=0.5, num_classes=num_classes)

def densenet_169(num_classes=1000):
    return DenseNet([6, 12, 32, 32], k=32, theta=0.5, num_classes=num_classes)

def densenet_201(num_classes=1000):
    return DenseNet([6, 12, 48, 32], k=32, theta=0.5, num_classes=num_classes)

def densenet_264(num_classes=1000):
    return DenseNet([6, 12, 64, 48], k=32, theta=0.5, num_classes=num_classes)

if __name__ == '__main__':
    net1 = densenet_121(10)
    # print(net)
    # summary 一个方便打印网络结构模块
    summary(net1,(3,224,224))
    input = torch.randn(1,3,224,224)
    out = net1(input)
    print(out.shape)




