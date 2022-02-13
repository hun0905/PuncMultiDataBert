from pickle import TRUE
from sympy import Segment
from torch.nn.functional import embedding
from torch.utils import data
from torch.utils.data import dataset
import torch
from sklearn.model_selection import KFold
import time
import numpy as np
from dataset import PuncDataset
from dataset import Collate
from model import BertPunc
from torchvision.transforms import ToTensor
from torch.utils.data import  DataLoader,random_split,SubsetRandomSampler
from sklearn.metrics import precision_recall_fscore_support as score
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert import BertTokenizer
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
K_fold = False #是否使用k折交叉驗
class Train():
    def __init__(self,dataset, model, criterion, optimizer,use_cuda,batch_size,epochs,scheduler,punc_path,load_path = 'punc_model',collate_fn = None,is_continue=False,\
                num_worker = 8,batch_size_times = 1,pin_memory = False,k = 10,with_l1= False,with_l2=False,l1_weight = 0,l2_weight=0,K_fold = False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        k = 10
        self.start = 0 #決定start 的epoch正常是0 但如果是載入model繼續訓練則會是過去所訓練到的epoch
        self.epochs = epochs #epochs的上限 最多訓練幾epoch
        self.dataset = dataset #資料 傳入的是PuncData的型態
        self.epoch = 0 #目前所到的epcoh
        self.model = model #所選的model 可能是 TBRNN或bi-lstm
        self.criterion = criterion #所用的loss function 這裡是crossentropy
        self.optimizer = optimizer #優化器，用以調整參數，這裡用adam
        self.use_cuda = use_cuda #是否使用gpu
        self.batch_size = batch_size #batch_size大小
        self.scheduler = scheduler #學習率 decay 這裡用exp
        self.load_path = load_path #如果要繼續訓練model,指定之前的model所在的path
        self.collate_fn = collate_fn 
        self.is_continue = is_continue #重新訓練一個model或是繼續訓練之前的model
        self.num_worker = num_worker #Dataloader的參數，正常時不用刻意調整
        self.batch_size_times = batch_size_times #batch_size*batch_size_times就是真正的batch_size大小，因為gpu不足batch_size不能太大，要更大的batch_size時調整
        self.pin_memory = pin_memory #Dataloader的參數,不用調整
        self.with_l1 = with_l1 #是否添加l1正則化
        self.with_l2 = with_l2 #是否添加l2正則化
        self.l1_weight = l1_weight #l1正則化的weight
        self.l2_weight = l2_weight #l2正則化的weight
        self.K_fold = K_fold #是否使用k折,如果不用,則epoch 全執行完就跳出
        self.k = k #如果用k折,k的折數是多少
        with open(punc_path, encoding='utf-8') as file: 
            self.punc2id = { i + 1 : word.strip()for i, word in enumerate(file) } #建立index對punctuation的字典
        self.punc2id[0] = " " #沒有標點
        self.history_train_loss = [] #存每個epoch的train loss
        self.history_val_loss = [] #存每個epoch的 val loss
        
        if is_continue: #是否是繼續訓練舊model
            #載入舊model的狀態和各種參數
            package = torch.load(load_path)
            self.model = self.model.load_model(load_path).cuda()
            for p in self.model.bert.parameters():
                p.requires_grad = True
            #self.optimizer.load_state_dict(package['optim_dict'])
            self.scheduler.load_state_dict(package['scheduler'])
            self.start = package['epoch']
            self.history_train_loss = package['train_loss']
            self.history_val_loss = package['val_loss']
        torch.manual_seed(42) 
        self.splits=KFold(n_splits=k,shuffle=True,random_state=42) #將整個train dataset隨機分k份 ,k-1用來train , 一份用來validation
    def prfs(self,train_trues,train_preds,total_loss): #計算和評估各種指標並輸出
        precision, recall, fscore, support = score(train_trues, train_preds)#將label和predict比較，計算出各類別Precision,Recall,和F-score
        accuracy = accuracy_score(train_trues, train_preds) #計算全部的accuracy,包含空白
        print("Multi-class accuracy: %.2f" % accuracy) #accuracy 的精確度
        SPLIT = "-"*(12*4+3) #分隔線
        print(SPLIT)#分隔線輸出
        #f = lambda x : round(x, 2)
        #輸出每個標點符號的各種指標評估結果
        for (v, k) in sorted(self.punc2id.items(), key=lambda x:x[1]):
            if v >= len(precision): continue
            if k == " ":
                k = "  "
                continue
            print("Punctuation: {} Precision: {:.3f} Recall: {:.3f} F-Score: {:.3f}".format(k,precision[v],recall[v],fscore[v]))
        print(SPLIT)

        #計算和印出overall(總和不分類別)的所有指標
        sklearn_accuracy = accuracy_score(train_trues, train_preds) 
        sklearn_precision = precision_score(train_trues, train_preds, average='micro')
        sklearn_recall = recall_score(train_trues, train_preds, average='micro')
        sklearn_f1 = f1_score(train_trues, train_preds, average='micro')
        print("[sklearn_metrics] Total Epoch:{} loss:{:.4f} accuracy:{:.4f} precision:{:.4f} recall:{:.4f} f1:{:.4f}".format(self.epoch+1, \
            total_loss, sklearn_accuracy, sklearn_precision, sklearn_recall, sklearn_f1))
    def train_epoch(self,data_loader):
        self.model.train()  #確保layers of model 在train mode
        total_loss = 0
        train_preds = [] #存放model預估的標點
        train_trues = [] #存放label的真實標點
        for  i,(data) in enumerate(data_loader):
            #print(i)
            input ,segment, label = data#輸入的資料(文字換成index),句子長度，label(標點的index)
            #print(segment)
            if  self.use_cuda:
                input = input.cuda()
                label = label.cuda()
                segment = segment.cuda()
                input = input.to(self.device)
                label = label.to(self.device)
                segment = segment.to(self.device)
            #print(label)
            outputs = self.model(input,segment)#將資料輸入model(調用model的forward),outputs為評估結果
            #將outputs和label的dimension轉換，在用crossentropy評估loss
            #print(outputs.size())
            outputs = outputs.view(-1, outputs.size(-1))
            #print(outputs.size())
            if self.use_cuda:
                outputs = outputs.to(self.device)
            label = label.view(-1)
            loss = self.criterion(outputs, label)
            loss_with_reg = loss#loss_with_reg是有加入正則化的loss，如果沒加就和loss相等
            if self.use_cuda:
                loss_with_reg = loss_with_reg.to(self.device)
            if self.with_l1: #l1正則化
                l1 = 0
                l1 += sum ( [p.abs().sum() for p in self.model.encoder.parameters()] )
                l1 += sum ( [p.abs().sum() for p in self.model.decoder.parameters()] )
                l1 += sum ( [p.abs().sum() for p in self.model.projected.parameters()] )
                l1_penalty = self.l1_weight *l1
                loss_with_reg += l1_penalty
            if self.with_l2: #l2正則化
                l2 = 1e-3
                l2 += sum ( [(p**2).sum() for p in self.model.encoder.parameters()] )
                l2 += sum ( [(p**2).sum() for p in self.model.decoder.parameters()] )
                l2 += sum ( [(p**2).sum() for p in self.model.projected.parameters()] )
                l2_penalty = self.l2_weight *l2
                loss_with_reg += l2_penalty
            loss_with_reg.backward()#更新梯度
            clipping_value = 2 # arbitrary value of your choosing
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clipping_value)
            if (i+1) % self.batch_size_times == 0 or (i+1) == len(data_loader):
                self.optimizer.step() #計算weight
                self.optimizer.zero_grad() #將梯度清空
            total_loss += loss.item()
            train_outputs = outputs.argmax(dim=1) #outputs原輸出的是四種class的機率分佈,換成最高機率class的index
            #print('train: ',train_outputs)
            #print('label',label)
            train_preds.extend(train_outputs.detach().cpu().numpy())
            train_trues.extend(label.detach().cpu().numpy())
            
        #if self.scheduler.get_last_lr()[0] > 1.5e-4:
        self.scheduler.step()#進行learning rate decay
        print('train: ','\n')
        self.prfs(train_trues,train_preds,total_loss)#印出這個epoch的train的結果評估
        return total_loss/(i+1)
    def val_epoch(self,data_loader):
        val_loss = 0
        self.model.eval()#告訴model不要學新東西

        #後面大致跟train epoch差不多

        val_preds = []
        val_trues = []
        for i,(data) in enumerate(data_loader):
            input , segment , label = data
            if  self.use_cuda:
                input = input.cuda()#換成可傳入gpu的型態
                label = label.cuda()
                segment = segment.cuda()
                input = input.to(self.device)
                label = label.to(self.device)
                segment = segment.to(self.device)
            
            outputs = self.model(input,segment)
            outputs = outputs.view(-1, outputs.size(-1))
            if self.use_cuda:
                outputs = outputs.to(self.device)
            label = label.view(-1)
            loss = self.criterion(outputs, label)
            val_loss += loss.item()
            val_outputs = outputs.argmax(dim=1)

            val_preds.extend(val_outputs.detach().cpu().numpy())
            val_trues.extend(label.detach().cpu().numpy())
        print("validation: ",'\n') 
        self.prfs(val_trues,val_preds,val_loss) #印出valdation 結果的評估
        return val_loss/(i+1)
    def train(self):
        for fold, (train_idx,val_idx) in enumerate(self.splits.split(np.arange(len(self.dataset)))):
            #train_idx 是被選為train data的資料的idx val_idx 是 val_data的資料的idx
            train_sampler = SubsetRandomSampler(train_idx)#定義train的取樣方式，決定train要取哪些資料
            val_sampler = SubsetRandomSampler(val_idx)#決定val要取哪些資料
            train_loader = DataLoader(self.dataset, batch_size=self.batch_size, sampler=train_sampler,collate_fn=self.collate_fn,num_workers=self.num_worker,pin_memory=self.pin_memory)
            val_loader = DataLoader(self.dataset, batch_size=self.batch_size, sampler=val_sampler,collate_fn=self.collate_fn,num_workers=self.num_worker,pin_memory=self.pin_memory)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            #有沒有attention 的model檔名開頭不同
           
            path = '/home/yunghuan/Desktop/PuncBert/model/Bert.pth.tar'

            print(device,'\n')

            #開始進入epoch，每一個epoch 都會經歷train epoch和val epoch
            for epoch in range(self.start,self.epochs):
                self.epoch = epoch
                train_loss=self.train_epoch(train_loader)#回傳train loss
                val_loss=self.val_epoch(val_loader)#回傳val loss
                print(f"Epoch:{self.epoch + 1} ; {self.epochs} average Training Loss:{train_loss} ; average Val Loss:{val_loss} ")
                self.history_train_loss.append(train_loss)#將train loss 存入list
                self.history_val_loss.append(val_loss) #將val loss 存入 list

                #在checkpoint 將model儲存起來 要存取model和optimizer的state和epoch,history train及val loss語各種hyperparameter
                #詳細部分可看Seq2Seq model 的 serialize
                torch.save( self.model.serialize(self.model,self.optimizer,self.scheduler,epoch,self.history_train_loss,self.history_val_loss)\
                             ,path+str(self.epoch+1))   

            if self.K_fold == False: #如果沒有k fold在這裡就會停止
                break
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) #計算model用了多少參數，可以大約估算model的大小
def main():

    '''
    language是語言選擇 zh是中文 , Eng是英文
    vocab_path 是 vocabulary的路徑
    punc_path 是 標點符號字典的路徑
    data_path 是 train_data的路徑
    input_size 是設定model中embedding層輸入的size,是vocab數量特殊符號數量的合，就是vocabulary詞彙數量加4
    hidden_dim就是定義model中的rnn 隱藏層的輸出維度
    embedding_dim就是nn.Embedding 的輸出維度，如果要調用pretrained的話必須為300
    '''

    language = 'zh'
    if language == 'Eng':
        vocab_path = '/home/yunghuan/NLP_Dataset/English/data_Eng/vocabulary'
        punc_path = '/home/yunghuan/NLP_Dataset/English/data_Eng/punc_vocab'
        data_path = '/home/yunghuan/NLP_Dataset/English/data_Eng/train.txt'
        input_size = 27180
    elif language == 'zh':
        vocab_path = '/home/yunghuan/NLP_Dataset/Chinese/data_Ch/vocab.txt'
        punc_path = '/home/yunghuan/NLP_Dataset/Chinese/data_Ch/punc.txt'
        data_path = '/home/yunghuan/Desktop/NLP_dataset_high/dataCutForBert/formosaTrain.csv'
        input_size = 37562


    dataset = PuncDataset(data_path,punc_path)#指定所用的dataset為PuncDataset

    '''collate_fn 就是將調用 dataset中的getitem 所得到的資料進行拼接以我們要的形式輸出
    也就是在 for i,(data) in enumerate(data_loader) 當中 data所得到的資料就是拼接後的結果'''
    collate_fn =Collate()
    

    embedding_dim = 300
    hidden_dim = 300
    
    '''
    如果是with attention mechanism機制，則要先定義encoder和decoder在把他們傳入Seq2Seq的model
    簡單說所要的的就是先使用encoder把我們的資料進行的編碼，把它們編成特定的資料型態，然後將我們所得到的
    編碼(rnn的 output也就就是最後一層的輸出) 傳是decoder,在decoder會將rnn的output傳入另一個rnn解碼，
    再將每個時刻的輸出和所有時刻輸出的序列進行attention,得到最後的解，詳細參考model2
    而如果沒有with attention 則就是直接使用bi-lstm即可
    '''

    '''
    input_size,embedding_size都如之前所定義，而output_size就是最後輸出的維度（標點數量＋1），num_layers是rnn的層術
    ,p 是dropout,pretrained是embedding是否要用pretrained word vector

    encoder(input_size,embedding_size,hidden_size,output_size,num_layers,p,pre_trained)
    decoder(hidden_size,output_size,num_layers,p)
    Seq2Seq(encoder,decoder,hidden_size)
    bi_LSTM(input_size,embedding_dim,hidden_size,num_layers,output_size,pretrained = True)
    '''
    save_path = '/home/yunghuan/Desktop/PuncBert/model/BertWiki.pth.tar6'
    model =BertPunc(384,4,0.3)
    for p in model.bert.parameters():
        p.requires_grad = True

    use_cuda = True
    if use_cuda:
        model = model.cuda()
    print(model)
    print('parameters_count:',count_parameters(model))
    criterion = torch.nn.CrossEntropyLoss(ignore_index=4) #決定loss function
    optimizer = torch.optim.Adam(model.parameters(),lr = 3e-7,weight_decay=0.0)#決定optimizer （更新weight 的方式）
    scheduler = ExponentialLR(optimizer, gamma=1,verbose = True) # weight decay的方式（非必要）

    '''
    Train(dataset, model, criterion, optimizer,use_cuda,batch_size,epochs,scheduler,punc_path,load_path = 'punc_model',collate_fn = None
    ,is_continue=False,num_worker = 8,batch_size_times = 1,pin_memory = False,k = 10,with_l1= False,with_l2=False,l1_weight = 0,l2_weight=0
    
    dataset： 資料 傳入的是PuncData的型態
    model： 所選的model 可能是 TBRNN或bi-lstm
    criterion： 所用的loss function 這裡是crossentropy
    use_cuda： 是否使用gpu
    batch_size： batch_size大小
    epochs： epochs的上限 最多訓練幾epoch
    scheduler： 學習率 decay 這裡用exp 
    punc_path : 標點符號字典路徑
    collate_fn : 選擇的collate_fn方式 此處用維我們自定義的collate_fn詳見dataset
    is_continue : 重新訓練一個model或是繼續訓練之前的model
    num_worker : Dataloader的參數，正常時不用刻意調整
    batch_size_times : batch_size*batch_size_times就是真正的batch_size大小，因為gpu不足batch_size不能太大，要更大的batch_size時調整
    pin_memory : Dataloader的參數,不用調整
    k #如果用k折,k的折數是多少
    with_l1 : 是否添加l1正則化
    with_l2 : 是否添加l2正則化
    l1_weight : l1正則化的weight
    l2_weight : l2正則化的weight
    '''

    TBRNN = Train(dataset,model,criterion,optimizer,use_cuda,1,30,scheduler,punc_path,save_path,collate_fn,True,batch_size_times=8,num_worker=0)
    TBRNN.train()
if __name__ == '__main__':
    main()

#5e-6