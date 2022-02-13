from opencc import OpenCC
import os
import json
import re
import jieba

dir = '/home/yunghuan/Desktop/Punctuation_test_now/PuncBiLstm_924/embedding_pretrained/corpus.zhwiki.txt'
#out_path = '/home/yunghuan/Desktop/Punctuation_test_now/PuncBiLstm_924/embedding_pretrained/corpus_cut.txt'
file = open(dir,'r',encoding='utf-8').readlines()
out_file = open(out_path,'w',encoding='utf-8')
cc = OpenCC('s2tw')
for line in file:
    sen = cc.convert(line)
    sen = ' '.join( jieba.cut(sen,cut_all=False,HMM=False) )
    out_file.write(sen)
    print(sen)
