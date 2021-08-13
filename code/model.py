# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 08:40:45 2021

@author: 莱克斯
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class Mydata(Dataset):
    def __init__(self,data,label):
        data=data
        label=label
        self.x = list(zip(data,label))
    def __getitem__(self, idx):
        
        assert idx < len(self.x)
        return self.x[idx]
    def __len__(self):
        
        return len(self.x)
    


class MLP(nn.Module): # line 1
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__() # line 3
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size,num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class attention_MLP(nn.Module): # line 1
    def __init__(self, att_size,input_size, hidden_size, num_classes):
      super().__init__() # line 3
      self.attention = nn.Linear(att_size,1)
      self.softmax = nn.Softmax(dim=1)
      self.fc1= nn.Linear(input_size,hidden_size)
      self.relu = nn.ReLU()
      self.fc2 = nn.Linear(hidden_size,num_classes)

    def forward(self, k,v):
      score = self.attention(k)
      alpha = self.softmax(score)
      inputs=(v*alpha).sum(axis=0)
      out = self.fc1(inputs)
      out = self.relu(out)
      out = self.fc2(out)
      return out
  
    
USE_GPU=False
def create_tensor(tensor):  # 是否使用GPU
    if USE_GPU:
        device = torch.device("cuda:0")
        tensor = tensor.to(device)
    return tensor 

# 2：构造模型
class GRU(nn.Module):
    """
    这里的bidirectional就是GRU是不是双向的，双向的意思就是既考虑过去的影响，也考虑未来的影响（如一个句子）
    具体而言：正向hf_n=w[hf_{n-1}, x_n]^T,反向hb_0,最后的h_n=[hb_0, hf_n],方括号里的逗号表示concat。
    """
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, bidirectional=True):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = 2 if bidirectional else 1 # 双向2、单向1
        
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,  # 输入维度、输出维度、层数、bidirectional用来说明是单向还是双向
                         batch_first=True,
                         bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size * self.n_directions, output_size)
        
    def __init__hidden(self, batch_size):  # 工具函数，作用是创建初始的隐藏层h0
        hidden = torch.zeros(self.n_layers * self.n_directions,
                            batch_size, self.hidden_size)
        return create_tensor(hidden) # 加载GPU
    
    def forward(self, input):
        # input shape:B * S -> S * B
        
        batch_size = input.size(0)
        
        hidden = self.__init__hidden(batch_size) # 隐藏层h0
        
        output, hidden = self.gru(input, hidden) # 只需要hidden
        if self.n_directions == 2: #双向的，则需要拼接起来
            hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden_cat = hidden[-1] # 单向的，则不用处理
        
        fc_output = self.fc(hidden_cat) # 最后来个全连接层,确保层想要的维度（类别数）
        return fc_output
    
    
    
  
    
  