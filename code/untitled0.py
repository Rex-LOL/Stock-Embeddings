# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 13:19:54 2021

@author: 莱克斯
"""
import re

with open('../data/corpus/Reuter&bloom_corpus.txt','rb') as fi:
    content=fi.readlines()

def remove_punctuation(line):
    rule = re.compile("[^a-zA-Z0-9\u4e00-\u9fa5]")
    line = rule.sub(' ',line)
    return line
dic = {}
for sentence in content:
    words=remove_punctuation(sentence.decode()).lower().split()
    for word in words:
        if word not in dic:
            dic[word] = 1
        else:
            dic[word] = dic[word] + 1

