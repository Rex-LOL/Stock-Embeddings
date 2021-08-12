
## Stock Embeddings Acquired from News Articles and Price History, andan Application to Portfolio Optimization
 论文复现 [《Stock Embeddings Acquired from News Articles and Price History, andan Application to Portfolio Optimization》](https://aclanthology.org/2020.acl-main.307/)
 ---
## 运行环境

Python 3.8 & Pytorch 1.9.0
---
## 数据描述

+ #### 新闻文本数据<br>
来自于哈工大SCIR开源的路透(106521条)&彭博(444958条)新闻数据集，时间范围为2006.10.20——2013.11.20，未标注单条新闻归属于哪只个股。提取了新闻标题至data/news_title.csv<br><br>
由于版权原因，本项目中未包含完整数据集，运行项目首先需下载[数据集](https://drive.google.com/drive/folders/0B3C8GEFwm08QY3AySmE2Z1daaUE?resourcekey=0-pbrVOqwKjQj3wRoCU8LiCA)并解压至项目文件夹data下
+ #### 股票价格数据<br>

选择了20只标普500的成分股作为实验对象，选择方法为在新闻文本中股票代码被提及1000次及以上的股票，股票价格数据来自于[英为财情](https://cn.investing.com/)，存放至data/stock price下<br><br>
对每只股票逐日计算对数收益率，并设置阈值，认为在（−0.0059,0.0068）间的股票涨跌情况是模糊的，将之剔除，因此，通过仅使用明显的负面和正面天数，回报被二值化<br><br>
以APPLE为例，在新闻覆盖时间内，共经历了1784个交易日，其中包含上涨656天、下跌592天、模糊536天，合计涨幅为652.13%
---
## 代码文件
* ### corpus_gen.py：包含了语料库生成及词向量训练过程，需在文本特征蒸馏前完成运行并保存结果
* ### text_feature_distiller.py ：提取文本特征，生成N_k(64维)及N_v(256维)
* ### tool.py : 包含了数个工具方法，其中重要方法有
    * `get_sj()` 生成并训练股票embedding，参数：train 训练集 test 测试集，默认为None DVR 是否用双向量表示，默认为Ture EVAL 是否开启验证，默认为False
    * `get_mj()` 生成市场向量，参数：weighted 简单平均或者加权平均,默认为False Sj 若加权平均，传递Sj以计算权重
    * `padding()` 对新闻数量较少的日用0向量进行填充，以输入attention-MLP对Stock Embeddings进行训练，参数：max_len 每日最大新闻数 save 是否保存填充结果至pickle文件
    * `train_test_split()` &nbsp;&nbsp;集成了股票数据读取、收益二值化、数据集划分，参数：price_path 股票文件路径
* ### model.py 包含用于训练Stock Embeddings的Attention-Mlp及用于股票运动分类的Bi-GRU
* ### shared_classifier.py 共享分类器机制的实现
* ### run.py 单一股票的模型训练全过程

---
## 待完善
* ~~双向量表示~~
* ~~加权平均市场向量~~
* ~~共享分类器~~
* 模型初始化固定，调参

