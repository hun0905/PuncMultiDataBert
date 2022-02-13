from opencc import OpenCC
import os
import json
import re
import jieba
from gensim.models.fasttext import FastText
dir = '/home/yunghuan/Desktop/Punctuation_test_now/PuncBiLstm_924/embedding_pretrained/corpus_cut.txt'
#out_path = '/home/yunghuan/Desktop/Punctuation_test_now/PuncBiLstm_924/embedding_pretrained/corpus_nopunc.txt'
file = open(dir,'r',encoding='utf-8').readlines()
wiki_sentences = []
for i in file:
    wiki_sentences.append(i.split())
model = FastText(wiki_sentences, sg=1, hs=1, size=300, workers=12, iter=5, min_count=10)
model.save('FastText.model')