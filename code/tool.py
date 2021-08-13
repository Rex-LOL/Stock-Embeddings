# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 08:54:43 2021

@author: 莱克斯
"""
import numpy as np
import pickle
import pandas as pd
from model import attention_MLP
import torch
import torch.nn as nn

def softmax(x, axis=1):
    # 计算每行的最大值
    row_max = x.max(axis=axis)

    # 每行元素都需要减去对应的最大值，否则求exp(x)会溢出，导致inf情况
    row_max=row_max.reshape(-1, 1)
    x = x - row_max

    # 计算e的指数次幂
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)
    s = x_exp / x_sum
    return s


#在通过attention训练stock embeddings时，输入的序列应保持同一长度，故对新闻数较少的日以0向量进行填充
def padding(max_len,save=True):
    
    with open(r"../data/embeddings_pca.pkl", 'rb') as fi:
        n_v = pickle.load(fi) 
        
    with open("../data/N_k.pkl", 'rb') as fi:
        n_k = pickle.load(fi)
        
    news_data=pd.read_csv('../data/news_title.csv',index_col=0)
    news_data.day=news_data.day.apply(lambda x: x.replace('-0','-'))
    
    n_v_list=[]
    n_k_list=[]
    
    news_num_day=news_data.groupby('day').count().news.to_list()
    
    
    flag=0
    for i in news_num_day:
        gap=max_len-i if i<=max_len else 0
        
        i_day_index_low=flag
        i_day_index_high=i_day_index_low+ (i if i<=max_len else max_len)
        
        flag=flag+i
        index_list=list(range(i_day_index_low,i_day_index_high))
        
        n_v_day=np.append(n_v[index_list],np.zeros((gap,256)),axis=0)
        n_k_day=np.append(n_k[index_list],np.zeros((gap,64)),axis=0)
        
        n_v_list.append(n_v_day)
        n_k_list.append(n_k_day)
        
    del n_v
    del n_k
    
    n_k_pad=np.array(n_k_list)
    
    del n_k_list
    
    n_v_pad=np.array(n_v_list)
    
    del n_v_list
    
    if save:
        #存储结果为pickle
        with open("../data/n_k_pad.pkl", 'wb') as fo:
            pickle.dump(n_k_pad, fo)
        
        del n_k_pad
            
        with open("../data/n_v_pad.pkl", 'wb') as fo:
            pickle.dump(n_v_pad, fo)
        
        del n_v_pad
    
        print(f'填充完毕，每日最大新闻数量为{max_len},已存储为pickle')
    else:
        return n_k_pad,n_v_pad

def convert_to_label(rf_amount):
    if rf_amount>0.0068:
        return 1
    elif rf_amount<-0.0059:
        return 0
    else:
        return 999

def train_test_split(price_path):
    
    test_year_list=['2013','2012']
    
    price_df=pd.read_csv(price_path)
    price_df.日期=price_df.日期.apply(lambda x:x.replace('年','-').replace('月','-').replace('日',''))
    
    price_df.涨跌幅=price_df.涨跌幅.apply(lambda x:float(x.replace('%','')))
    price_df.涨跌幅=price_df.涨跌幅.apply(lambda x:np.log(x/100+1))
    price_df['label']=price_df.涨跌幅.apply(convert_to_label)
    price_df=price_df[price_df.label!=999]
    
    
    news_data=pd.read_csv('../data/news_title.csv',index_col=0)
    news_data.day=news_data.day.apply(lambda x: x.replace('-0','-'))
    day_list=list(news_data.day.drop_duplicates())
    
    train=price_df[price_df.日期.apply(lambda x : all(year not in x for year in test_year_list))]
    test=price_df[price_df.日期.apply(lambda x : any(year in x for year in test_year_list))]
    
    train.insert(1,'index',train.日期.apply(lambda x : day_list.index(x)))
    test.insert(1,'index',test.日期.apply(lambda x : day_list.index(x)))

    
    return train[['index','label']],test[['index','label']]




def get_sj(train_df,test_df=None,eval_=False,DVP=True):
    with open("../data/n_k_pad.pkl", 'rb') as fi:
        n_k_pad=pickle.load(fi)
    
    with open("../data/n_v_pad.pkl", 'rb') as fi:
        n_v_pad=pickle.load(fi)

    model=attention_MLP(64,256,256,2)
    
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(10):
        for index,row in train_df.iterrows():
            train_index=row['index']
            label=torch.tensor(row['label'],dtype=torch.long).reshape(1)
            
            # 前向传播
            k=torch.tensor(n_k_pad[train_index],dtype=torch.float32)
            v=torch.tensor(n_v_pad[train_index],dtype=torch.float32)
            outputs = model(k,v)
            loss = criterion(outputs.reshape(1,2), label)
        
            # 后向优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        #print(epoch+1,'epoch loss:',loss.item())
    
    S_j=model.attention.weight.detach().numpy().reshape(64,1)
    
    if eval_ and test_df is not None:
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            
            for index,row in test_df.iterrows():
                test_index=row['index']
                label=torch.tensor(row['label'],dtype=torch.long).reshape(1)
                
                k=torch.tensor(n_k_pad[test_index],dtype=torch.float32)
                v=torch.tensor(n_v_pad[test_index],dtype=torch.float32)
        
                outputs = model(k,v)
                
                pred = outputs.argmax(dim = 0)
                total += label.size(0)
                correct += (pred == label).sum().item()
        
            print('The Accuracy of attention-mlp on test dataset: {} %'.format(100 * correct / total))
        acc=correct / total
        return S_j,acc
    else:
        return S_j


def get_mj(s_j=None,weighted=False):
    news_data=pd.read_csv('..//data//news_title.csv',index_col=0)
    news_data.day=news_data.day.apply(lambda x: x.replace('-0','-'))
    
    date_index_dict={}
    count=0
    for i in news_data.day:
        if i in date_index_dict.keys():
            date_index_dict[i].append(count)
        else:
            date_index_dict[i]=[count]
        count+=1
    
    with open(r"../data/embeddings_pca.pkl", 'rb') as fi:
        n_v = pickle.load(fi) 
        
    with open("../data/N_k.pkl", 'rb') as fi:
        n_k = pickle.load(fi)
    
    if s_j is not None and weighted:        
        score=np.dot(n_k,s_j)
    m_list=[]
    
    for index_list in date_index_dict.values():

        n_v_t = n_v[index_list]
        
        #Weighted Average
        if s_j is not None and weighted:  
            score_t = score[index_list]
            a_i_j=softmax(score_t.reshape(1,-1))
            m_j_t=np.multiply(n_v_t, a_i_j.reshape(-1,1)).sum(axis=0)
        else:
        #Simple Average
            m_j_t=n_v_t.sum(axis=0)
            
        m_list.append(m_j_t)
    
    m_j=np.array(m_list)
    m_j[np.isnan(m_j)]=0
    
    return m_j

def gru_data(df,m_j):
    x=[]
    y=[]
    for index,row in df.iterrows():
        index_list=list(range(row['index']-4,row['index']+1))
        y.append(row['label'])
        x.append(m_j[index_list])
    return x,y



