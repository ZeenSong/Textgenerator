#!/usr/bin/env python
# coding: utf-8

# In[1]:


import jieba
import re
import torch
import torch.nn as nn
import gensim
from torch.utils.data import Dataset,DataLoader
from gensim.corpora.dictionary import Dictionary
from torch.nn import functional as F

# path = './harrypotter.txt'
# with open(path,'r',encoding='gb18030') as f:
#     article = f.read()
#     f.close()

# def text2sentence(text):
# #     sentences = []
#     raw_sentence = re.findall("[\w+[\，\、\"]*\w+[\！\。\？]\"*]*",text)
# #     for sentence in raw_sentence:
# #         sentences.append(''.join(re.findall("\w+",sentence)))
#     return raw_sentence



# corpus = list(jieba.cut(article))
# dictionary = Dictionary([corpus])
# stop_words = [' ' , '\u3000' , '\ue4c6' , '」' , '「' , '┅' , '\n','…']
# corpus = [word for word in corpus if word not in stop_words ]

# with open("./corpus.txt",'w',encoding='utf-8') as f:
#     f.write(" ".join(corpus))

sentences = gensim.models.word2vec.Text8Corpus("./text_clean.txt")

model=gensim.models.word2vec.Word2Vec(sentences,min_count=0,size=300)

model.wv.save_word2vec_format('HP'+'.model.bin',binary=True)

