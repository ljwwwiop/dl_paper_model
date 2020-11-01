import torch

'''
    主要是防止过拟合，增强模型的泛化能力，在one-hot基础上添加一个平滑系数&
    使得最大预测与其他类别平均值之间的差距的经验分布更加平滑
'''

def smooth_one_hot(true_labels,classes,smoothing = 0.0):
    '''
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method
    '''
    # 断言
    assert 0<= smoothing <1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0),classes))  # torch.Size([2, 5])
    with torch.no_grad():
        # 空的  没有
        true_dist = torch.empty(size=label_shape, device=true_labels.device)
        true_dist.fill_(smoothing/(classes-1))
        _,index = torch.max(true_labels,1)
        true_dist.scatter_(1, torch.LongTensor(index.unsqueeze(1)), confidence)  # 必须要torch.LongTensor()
    return true_dist

true_labels = torch.zeros(2, 5)
true_labels[0, 1], true_labels[1, 3],true_labels[1, 4] = 1, 1,0.5
print('标签平滑前:\n', true_labels)

true_dist = smooth_one_hot(true_labels, classes=5, smoothing=0.1)
print('标签平滑后:\n', true_dist)

