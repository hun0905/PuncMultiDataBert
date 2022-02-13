from nltk.util import pr
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.utils.data as data 
import nltk
from pytorch_pretrained_bert import BertTokenizer
def make_id(seqs,punc_vocab_path):
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)
    with open(punc_vocab_path, encoding='utf-8') as file:
        punc_vocab = { word.strip(): i + 1 for i, word in enumerate(file) } #製作標點字典
    punc_vocab[" "] = 0 #沒有標點
    seqs.sort(key=lambda x: len(x.split()), reverse=True) 
    seq_ids = []
    label_ids = []
    for i in range(len(seqs)):
        seqs[i] = seqs[i].replace(' ','')
        input = []
        label = []
        input.append(tokenizer.convert_tokens_to_ids(['[CLS]'])[0])
        label.append(punc_vocab[' '])
        punc = ' '
        for token in seqs[i]:
            if token in punc_vocab:
                punc = token
            else:
                tokens = tokenizer.tokenize(token)
                if not tokens:
                    continue
                #print(self.tokenizer.convert_tokens_to_ids(tokens))
                input.append(tokenizer.convert_tokens_to_ids(tokens)[0])
                label.append(punc_vocab[punc])
                punc = ' '
        input.append(tokenizer.convert_tokens_to_ids(['[SEP]'])[0])
        label.append(punc_vocab[punc])
        seq_ids.append(input.copy())
        label_ids.append(label.copy())
    return seq_ids,label_ids
class PuncDataset(data.Dataset):
    def __init__(self, file_path, punc_vocab_path):
        self.seqs = open(file_path, encoding='utf8',errors='ignore').readlines()

        self.seq_ids,self.label_ids = make_id(self.seqs,punc_vocab_path)
    def __len__(self):
        return len(self.seqs)
    def __getitem__(self, index):
        input = self.seq_ids[index]
        label = self.label_ids[index]
        input = torch.LongTensor(input)
        label = torch.LongTensor(label)
        return input,label
class NoPuncDataset(data.Dataset): #NoPuncDataset 是沒有標點符號的dataset 目的是要拿來做test
    def __init__(self, file_path, punc_vocab_path):
        self.seqs = open(file_path, encoding='utf8',errors='ignore').readlines()
        self.seq_ids,self.label_ids = make_id(self.seqs,punc_vocab_path)
    def __len__(self):
        return len(self.seqs)
    def __getitem__(self, index):
        seq = self.seqs[index].strip('\n')
        seq_id = self.seq_ids[index]
        seq_id = torch.LongTensor(seq_id)
        return seq_id,seq
    def _process(self,seq):
        """Convert txt sequence to word-id-seq."""
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)
        input = []
        input.append(tokenizer.convert_tokens_to_ids(['[CLS]'])[0])
        for word in seq.split():
            input.append(tokenizer.convert_tokens_to_ids(word)[0])
        input.append(tokenizer.convert_tokens_to_ids(['[SEP]'])[0])
        input = torch.LongTensor(input)
        return input

class Collate(object):
    def __init__(self):
        pass
    def __call__(self, batch):
        batch.sort(key=lambda x: len(x[0]), reverse=True)
        input_seqs, label_seqs = zip(*batch)
        lengths = [len(seq) for seq in input_seqs]
        #max_length = 100
        
        segments = [[0]*384]*len(input_seqs)
       
        #print(max_length)
        input_padded = []
        label_padded = []
        #print(len(input_seqs[0]))
        for i, (input, label) in enumerate(zip(input_seqs, label_seqs)):
            n = 384-lengths[i]
            input_padded.append( list( nltk.pad_sequence(input[:384],n+1,pad_right=True,right_pad_symbol=0) )  ) 
            label_padded.append( list( nltk.pad_sequence(label[:384],n+1,pad_right=True,right_pad_symbol=4) )  ) #4 is end
            #print(torch.IntTensor(lengths))
        input_padded = torch.tensor(input_padded)
        label_padded = torch.tensor(label_padded)
        
        return input_padded, torch.tensor(segments), label_padded

