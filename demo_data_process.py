import torch
import copy



def process(file_path='data/test.txt',\
            punc_vocab_path='data/punc_vocab',\
            out_path = 'data_after_process/demo.txt'):
    with open(punc_vocab_path, encoding='utf-8') as file:
        punc_vocab = { word.strip(): i + 1 for i, word in enumerate(file) } 
    punc_vocab[" "] = 0 #沒有標點
    print(punc_vocab)
    seqs = open(file_path, encoding='utf8',errors='ignore').read()
    tmp = []
    sentence = []
    end = True
    for i,word in enumerate(seqs.split()):
        if word not in punc_vocab.keys() and end == True:
            tmp.append(word)
        if (i+1) % 99 == 0:
            if word != '.PERIOD' and word !='?QUESTIONMARK' :
                end = False
            sentence.append(copy.copy(tmp))
            tmp.clear()
        if word == '.PERIOD' or word =='?QUESTIONMARK' :
            end = True
    out_file = open(out_path,'w', encoding='utf8',errors='ignore')
    for i in sentence:
        out_file.write(' '.join(i))
        out_file.write('\n')
            
        
    



def main():
    process()
if __name__ == '__main__':
    main()