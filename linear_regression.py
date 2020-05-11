import torch
import numpy as np 
from torch import nn
import random 
import torch.utils.data  as Data 
#pytorch 数据都为tensor类型
# 生成数据集
num_inputs = 2
num_samples = 1000
true_w = torch.tensor([3.14,-4])
true_b = torch.tensor([-2])
features = torch.tensor(np.random.normal(0,1,(num_samples,num_inputs)),dtype=torch.float32)
labels = torch.matmul(features,true_w) + true_b
noise = torch.tensor(np.random.normal(0,0.1,size=labels.size()),dtype=torch.float32)
labels = torch.add(labels,noise)

#读取数据
batch_size = 64
dataset = Data.TensorDataset(features,labels)
#随机读取小批量
data = Data.DataLoader(dataset,batch_size,shuffle=True)

# for X,y in data:
#     print(X,y)
#     break

#定义模型
class Net(nn.Module):
    def __init__(self,num_feature):
        super(Net, self).__init__()
        self.linear = nn.Linear(num_feature,1)

    def forward(self, x):
        out = self.linear(x)

        return out
net = Net(num_inputs)
# print(net)

#初始化模型参数
nn.init.normal_(net.linear.weight, mean=0,std=0.01)
nn.init.constant_(net.linear.bias,val=0)

#定义损失函数
loss = nn.MSELoss()

#定义优化算法
optimizer = torch.optim.SGD(net.parameters(),lr=0.01)

#训练模型
Epoch = 15
for i in range(Epoch):
    for X,y in data:
        l = loss(net(X),y.view(-1,1))
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    print('Epoch:{} | Loss:{} '.format(i+1,l.item()))
for parameters in net.parameters():
    print(parameters)

