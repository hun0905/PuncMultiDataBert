import torch
from torch.utils.data import dataset
from model import BertPunc
import nltk
from dataset import NoPuncDataset
from torch.utils.data import Dataset, DataLoader
vocab_path = '/home/yunghuan/NLP_Dataset/Chinese/data_Ch/vocab.txt'
with open(vocab_path, encoding='utf-8',errors='ignore') as file:
    vocab = { word.strip(): i + 4 for i, word in enumerate(file) } 
def punc_vocab(punc_vocab_path):
    with open(punc_vocab_path, encoding='utf-8') as file:
        punc_vocab = { i + 1 : word.strip()for i, word in enumerate(file) } 
    punc_vocab[0] = " " #沒有標點
    return punc_vocab

def add_punc(seq, predict, class2punc):    
    txt_with_punc = ""
    seq = [s for s in seq]
    print(len(seq))
    print(len(predict))
    for i, word in enumerate(seq):
        punc = class2punc[predict[i+1]][0]
        txt_with_punc += word if punc == " " else punc + word 
    punc = class2punc[predict[i + 2]][0]
    txt_with_punc += punc
    print(txt_with_punc)
    return txt_with_punc

def add_punctuation(use_cuda = True):
    model = BertPunc.load_model('/home/yunghuan/Desktop/PuncBert/model/model_finetuning.pth.tar11')
    model.eval()
    if use_cuda:
        model = model.cuda()
    demo_result = open('demo_result.txt','w')
    dataset = NoPuncDataset('/home/yunghuan/Desktop/PuncBert/test.txt',\
                            '/home/yunghuan/NLP_Dataset/Chinese/data_Ch/punc.txt')
        
    data_loader = DataLoader(dataset, batch_size=1)
    class2punc = punc_vocab('/home/yunghuan/NLP_Dataset/Chinese/data_Ch/punc.txt')
    
    with torch.no_grad():
        for i,(seq_id,seq) in enumerate(data_loader):
            n = 384-seq_id.size()[1]
            seq_id =  torch.tensor( [list( nltk.pad_sequence(seq_id[0][:384],n+1,pad_right=True,right_pad_symbol=0) )] )
            segments =torch.tensor( [[0]*384]*1 )
           
            #print(seq[0])
            if use_cuda:
                seq_id = seq_id.cuda()
                segments = segments.cuda()
            output = model(seq_id,segments)
            output = torch.argmax( output.squeeze(0) , 1)
            #print(output)
            output = output.data.cpu().numpy().tolist()
            out_text = add_punc(seq[0],output,class2punc)
            demo_result.write(out_text)
            demo_result.write('\n')
        
def main():
    add_punctuation()
if __name__ == '__main__':
    main()
