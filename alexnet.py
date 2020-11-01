'''
    学习pytorch 和 学习 alexnet
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

'''
f(x)=x*tf.math.tanh(tf.softplus(x))
mish = Mish()
x = torch.linspace(-10,10,1000)
y = mish(x)
 
plt.plot(x,y)
plt.grid()
plt.show()
'''
class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()
    def forward(self, x):
        return x *torch.tanh(F.softplus(x))

# 建立模型
class AlexNet(nn.Module):
    '''
    前两个模型没有进行修改网络，第二次模型效果还可以
    2020/10/08
    这一次准备修改卷积大小，将感受野变小,224x224 -> 可能会有更多的参数
    '''
    # 初始化
    def __init__(self,num_class = 5):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # nn.Conv2d(3,64,kernel_size=11,stride=4,padding=2),
            # nn.ReLU(inplace=True) ,#inplace:原地　　不创建新对象，直接对传入数据进行修改
            # nn.Conv2d(64,192,kernel_size=5,padding=2),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3,stride=2),

            nn.Conv2d(3,64,kernel_size=5,stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,192,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=6,stride=4),

            nn.Conv2d(192,384,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2)
        )

        # self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*13*13,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096,num_class),
        )

    # 向前传播
    def forward(self, x):
        x = self.features(x)
        #函数将张量x变形成一维向量形式，总特征数不变，为全连接层做准备
        #2163200
        # print(x.shape)

        x = x.view(x.size(0),256*13*13)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # print("x :",x)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    net = AlexNet()
    input = torch.randn(1,3,224,224)
    out = net(input)
    print(out)

