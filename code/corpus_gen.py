# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 19:44:58 2021

@author: 莱克斯
"""

import os

with open(r'..\data\corpus\Reuter&bloom_corpus.txt','wb') as corpus:
    path='..\data\ReutersNews106521'
    count=0
    for day in os.listdir(path):
        day_path=path+'\\'+day
        #print(forder_path)
        #break
        for news in os.listdir(day_path):
             news_path=day_path+'\\'+news
             # print(news_path)
             with open(news_path,'rb') as file:
                  content = list(map(lambda x: x.lower(),file.readlines()[7:]))
                  corpus.writelines(content)      
        #if count>10:break
        count+=1
        print(f'{day}日的新闻数据处理完毕')
    
    err_num=0
    path='..\data\20061020_20131126_bloomberg_news'
    count=0
    for day in os.listdir(path):
        day_path = path+'\\'+day
        for news in os.listdir(day_path):
            news_path=day_path+'\\'+news
            #print(news_path)
            try:
                with open(news_path,'rb') as file:
                  content = list(map(lambda x: x.lower(),file.readlines()[7:-14]))
                  corpus.writelines(content) 
            except(FileNotFoundError):
                err_num+=1
                continue
        #if count>10:break
        count+=1
        print(f'{day}日的新闻数据处理完毕')
        

#训练word2vec
import word2vec
word2vec.word2vec('../data/corpus/Reuter&bloom_corpus.txt', '../data/corpus/corpusWord2Vec_Reuter&bloom.bin', size=64,verbose=True)

from gensim.models import KeyedVectors
model = KeyedVectors.load_word2vec_format('corpusWord2Vec_Reuter&bloom.bin',binary=False, encoding="utf8",  unicode_errors='ignore')

import numpy as np
word2vec_dict = {}
for word, vector in zip(model.vocab, model.vectors):
  word2vec_dict[word] = vector /np.linalg.norm(vector) 
# for each in word2vec_dict:
#     print (each,word2vec_dict[each])

import pickle
file_name=input()
# with open("../data/embeddings_pca.pkl", 'wb') as fo:
#     pickle.dump(word2vec_dict, fo)
