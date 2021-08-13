# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 13:48:16 2021

@author: 莱克斯
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import GRU,Mydata
from tool import train_test_split,get_sj,get_mj,gru_data
import os

shared_train_x=[]
shared_train_y=[]
shared_test_x=[]
shared_test_y=[]

count=0

path='../data/stock price'
for stock in os.listdir(path):
    stock_path=path+'/'+stock
    print(stock_path)
    train,test=train_test_split(stock_path)
    S_j,acc=get_sj(train,test,eval_=True)
    
    
    if acc>0.51:
        count+=1
        M_j=get_mj(S_j,weighted=True)
        temp_train_x,temp_train_y=gru_data(train,M_j)
        temp_test_x,temp_test_y=gru_data(test,M_j)
        
        shared_train_x.extend(temp_train_x)
        shared_train_y.extend(temp_train_y)
        shared_test_x.extend(temp_test_x)
        shared_test_y.extend(temp_test_y)

for t in range(10):
    
    train_dataset=Mydata(shared_train_x,shared_train_y)
    test_dataset=Mydata(shared_test_x,shared_test_y)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    input_size = 256    #  256维市场向量
    hidden_size = 256   # 隐藏层大小
    num_classes = 2    # 涨跌二元判断
    num_epochs = 50  # 将num_epochs提高到50
    batch_size = 500    # 每一个batch的大小
    learning_rate = 0.001   # 学习率

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # 构造GRU模型
    model = GRU(input_size, hidden_size, num_classes).to(device)
    
    model.train()
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练模型
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (data, labels) in enumerate(train_loader):
          
            # Move tensors to the configured device
            data = data.reshape(-1, 5, 256).to(device).to(torch.float32)
            labels = labels.to(device)
      
            # 前向传播
            outputs = model(data)
            loss = criterion(outputs, labels)
      
            # 后向优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
  #测试模型
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for m_v, labels in test_loader:
            m_v = m_v.reshape(-1, 5 , 256).to(device)
            
            labels = labels.to(device)
            outputs = model(m_v.to(torch.float32))
            pred = outputs.argmax(dim = 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

    print('The {}th Accuracy of bi-gru on test dataset: {} %'.format(t+1,100 * correct / total))   