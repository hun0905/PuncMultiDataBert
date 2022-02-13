from nltk.util import pr
from sympy import Segment
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import torch.nn.functional as F
from pytorch_pretrained_bert import BertForMaskedLM
class BertPunc(nn.Module):
    def __init__(self, segment_size, output_size, p):
        super(BertPunc, self).__init__()
        self.bert = BertForMaskedLM.from_pretrained('/home/yunghuan/bert_model/chinese_rbt3_pytorch')
        self.segment_size = segment_size
        self.output_size = output_size
        self.bert_vocab_size = 21128
        self.bn = nn.BatchNorm1d(segment_size*self.bert_vocab_size)
        self.fc = nn.Linear(self.bert_vocab_size, output_size)
        self.p = p
        self.Dropout = nn.Dropout(p = self.p)
    def forward(self, x,segment):
        #print(x.size())
        #print(segment.size())
        x = self.bert(x,segment)
        #print(x.size())
        #print(x.size())
        #x = x.view(x.shape[0], -1)
        #x = self.bn(x)

        x = self.fc(self.Dropout(x))
        return x    
    @classmethod
    def load_model(cls, path):
        # Load to CPU
        package = torch.load(path, map_location=lambda storage, loc: storage)########
        model = cls.load_model_from_package(package)
        return model

    @classmethod
    def load_model_from_package(cls, package):
        model = cls(package['segment_size'], package['output_size'],
                    package['p'])
        model.load_state_dict(package['state_dict'])
        return model
        
    def serialize(self,model, optimizer,scheduler, epoch,train_loss,val_loss):
        package = {
            # hyper-parameter
            'segment_size': model.segment_size,
            'output_size': model.output_size,
            'p':model.p,
            # state
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'scheduler':scheduler.state_dict(),
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss':val_loss
        }
        return package
