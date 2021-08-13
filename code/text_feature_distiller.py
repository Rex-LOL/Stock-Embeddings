import pandas as pd
import numpy as np
import os
import pickle
import re
import time
from sklearn.decomposition import PCA

#仅保留新闻标题
'''
path='..\data\ReutersNews106521'
count=0
news_dict={}
news_dict['news']=[]
news_dict['day']=[]
for day in os.listdir(path):
    day_path=path+'\\'+day
    #print(forder_path)
    #break
    for news in os.listdir(day_path):
         news_path=day_path+'\\'+news
         #print(news_path)
         with open(news_path) as file:
             headline=file.readline()[3:]
             #print(headline)
             news_dict['news'].append(headline[:-1])
             news_dict['day'].append(day) 
    count+=1
    print(f'{day}日的新闻数据处理完毕')

news_data=pd.DataFrame(news_dict)
news_data['day']=news_data.day.apply(lambda x: str(x)[:4]+'-'+str(x)[4:6]+'-'+str(x)[6:])

path='..\\data\\20061020_20131126_bloomberg_news'
count=0
err_num=0
news_dict={}
news_dict['news']=[]
news_dict['day']=[]
for day in os.listdir(path):
    day_path=path+'\\'+day
    #print(forder_path)
    #break
    for news in os.listdir(day_path):
        news_path=day_path+'\\'+news
        #print(news_path)
        try:
            with open(news_path,'rb') as file:
                headline=file.readline()[3:]
                #print(headline)
                news_dict['news'].append(headline[:-1].decode())
                news_dict['day'].append(day) 
        except(FileNotFoundError):
            err_num+=1
            continue
    count+=1
    print(f'{day}日的新闻数据处理完毕')

news_data=pd.concat([news_data,pd.DataFrame(news_dict)])
news_data=news_data.dropna().drop_duplicates().sort_values('day').reset_index().drop('index',axis=1)
news_data.to_csv('..//data//news_title.csv')
'''

#td-idf 提取
'''
news_data=pd.read_csv('..//data//news_title.csv',index_col=0)

from sklearn.feature_extraction.text import CountVectorizer 

corpus=news_data.news.to_list()
  
#将文本中的词语转换为词频矩阵  
vectorizer = CountVectorizer()  
#计算个词语出现的次数  
X = vectorizer.fit_transform(corpus)  
#获取词袋中所有文本关键词  
word_list = vectorizer.get_feature_names()  
#print(word)  
#查看词频结果  
#print(X)

# ----------------------------------------------------

from sklearn.feature_extraction.text import TfidfTransformer  

#类调用  
transformer = TfidfTransformer()  
print (transformer)  
#将词频矩阵X统计成TF-IDF值  
tfidf = transformer.fit_transform(X)  
#查看数据结构 tfidf[i][j]表示i类文本中的tf-idf权重  
#print (tfidf)

#存储结果为pickle
with open("../data/word_count.pkl", 'wb') as fo:
    pickle.dump(X, fo)

with open("../data/tfidf.pkl", 'wb') as fo:
    pickle.dump(tfidf, fo)
    
with open("../data/word_list.pkl", 'wb') as fo:
    pickle.dump(word_list, fo)
'''


'''
#去除符号
def remove_punctuation(line):
    rule = re.compile("[^a-zA-Z0-9\u4e00-\u9fa5]")
    line = rule.sub('',line)
    return line

#读取词向量训练结果
with open(r"..\data\word2vec_dict_RB.pkl", 'rb') as fi:
    word2vec_dict = pickle.load(fi) 

#读取词频统计
with open("../data/word_count.pkl", 'rb') as fi:
    word_count = pickle.load(fi) 
    
#读取tf-idf权重值
with open("../data/tfidf.pkl", 'rb') as fi:
    tfidf = pickle.load(fi) 

#读取分词列表
with open("../data/word_list.pkl", 'rb') as fi:
    word_list = pickle.load(fi) 


#读取数据集股票新闻数据集
news_data=pd.read_csv('..//data//news_title.csv',index_col=0)
news_data.news=news_data.news.apply(lambda x: re.sub(r'[^\w\s]','',x))



#生成n_v,从文档的层次表示新闻文本
from sentence_transformers import SentenceTransformer

t1=time.time()
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
t2=time.time()

print(f'加载模型共耗时{t2-t1}')

embeddings=model.encode(news_data.news)

t3=time.time()

print(f'编码共耗时{t3-t2}')

#Print the embeddings
# for sentence, embedding in zip(sentences, embeddings):
#     print("Sentence:", sentence)
#     print("Embedding:", embedding)
#     print("")

pca=PCA(n_components=256)

embeddings_pca=pca.fit_transform(embeddings)

# with open("../data/embeddings_pca.pkl", 'wb') as fo:
#     pickle.dump(embeddings_pca, fo)



#n_k,从词的层次表示新闻文本
count=0
err_num=0
err_list=[]
n_k_list=[]
t1=time.time()
for a in range(word_count.shape[0]):
    weight_sum=0
    n_k_i=np.zeros(64)
    row=word_count.getrow(a)
    for b in range(word_count.getrow(a).count_nonzero()):
        
        try:
            x=a
            y=row.nonzero()[1][b]
            
            word=word_list[y]
            n_k_i_j=word2vec_dict[word]
            weight=tfidf[x,y]
            for n in range(word_count[x,y]):
                weight_sum+=weight
                n_k_i=n_k_i+n_k_i_j*weight

            #print(index)
        except(KeyError):
            err_list.append(word)
            err_num+=1
            continue
    n_k_list.append(n_k_i/weight_sum)
    count+=1
    if count%20000==0 and count>=20000:
        t2=time.time()
        print(t2-t1)
        print(count)
        t1=time.time()
        #break

# n_k=np.array(n_k_list)
# with open("../data/N_K.pkl", 'Wb') as fO:
#     pickle.DUMP(n_k,fi) 
#错误情况记录：词向量有14733个词缺失
'''







