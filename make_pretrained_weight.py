import torch

#製作vocabulary dictionary
file =  open('/home/yunghuan/NLP_Dataset/Chinese/data_Ch/vocab.txt','r', encoding='utf-8',errors='ignore').readlines()
word2idx = { word.strip(): i + 4 for i, word in enumerate(file) } 
idx2word = { i + 4: word.strip() for i, word in enumerate(file) }
word2idx['[PAD]'] = 0 
word2idx['[UNK]'] = 1 #UNK is unknown word
word2idx['[END]'] = 2 #END is the END of sequenc
word2idx['[CLS]'] = 3
idx2word[0] = '[PAD]'
idx2word[1] = '[UNK]'
idx2word[2] = '[END]'
idx2word[3] = '[CLS]'

#打開word vector的對照表
wv = open('/home/yunghuan/NLP_Dataset/Chinese/pretrained_vectors/zh_wiki_fasttext_300_ch.txt','r'\
        ,encoding='utf-8',errors='ignore').readlines()[1:]


#+4因為四種特殊符號
vocab_size = len(file)+4
embed_size = 300
weight = torch.zeros(vocab_size,embed_size)
count = 0

#找尋wv的每個單字 , 確認其是否有在vocabulary list有對應的單字 若有則將vector存入weight當中
for i in wv:
    line = i.split()
    try:
        index = word2idx[line[0] ]
        count+=1
    except:
        continue
    weight[index,:] = torch.FloatTensor(list(map(float,line[1:]))).cpu()
#定義特殊符號的word vector
weight[0,:] = torch.ones(300)
weight[1,:] = torch.zeros(300)
weight[2,:] = torch.full((300,),0.5)
weight[3,:] = torch.full((300,),2)
torch.save(weight,'/home/yunghuan/Desktop/PuncTBRNN/ChineseFastText.pth')
print(count)