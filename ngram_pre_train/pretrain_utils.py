import random
from transformers_down import BertTokenizer
import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
from itertools import chain
import pandas as pd


##
def loadData(path=None,data_type=None):
    if data_type=='unlabel':
        with open(path, 'r', encoding='utf-8') as f:
            words=[]
            for line in f:
                if '\t' in line:
                    words.extend(line.split('\t'))
                else:
                    words.append(line)

    if data_type=='train':

        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            lines.append('\n')

            words = []

            text = []

            for line in lines:

                if line == '\n':
                    text = ''.join(text)
                    if text == '':
                        continue
                    words.append(text)

                    text = []

                elif line == '  O\n':
                    text.append(' ')

                else:
                    line = line.strip('\n').split()
                    if len(line) == 1:
                        term = ' '

                    else:
                        term, label = line
                    text.append(term)


    # data=pd.DataFrame(words,columns=['words'])

    return words#list类型


##
def paddingList(ls:list,val,returnTensor=False):
    ls=ls[:]#不要改变了原list尺寸
    maxLen=max([len(i) for i in ls])
    for i in range(len(ls)):
        ls[i]=ls[i]+[val]*(maxLen-len(ls[i]))
    return torch.tensor(ls,device='cuda') if returnTensor else ls

def truncate(a:list,maxLen):
    maxLen-=2#空留给cls sep
    assert maxLen>=0
    #一共就a超长与否，b超长与否，组合的四种情况
    if len(a)>maxLen:#需要截断
            a=a[:maxLen]

    return a

class MLM_Data(Dataset):
    #传入句子对列表
    def __init__(self,textLs:list,maxLen:int,tk:BertTokenizer):
        super().__init__()
        self.data=textLs
        self.maxLen=maxLen
        self.tk=tk
        self.spNum=len(tk.all_special_tokens)
        self.tkNum=tk.vocab_size

    def __len__(self):
        return len(self.data)

    def random_mask(self,text_ids):
        input_ids, output_ids = [], []
        rands = np.random.random(len(text_ids))
        idx=0
        while idx<len(rands):
            if rands[idx]<0.15:#需要mask

                ngram=np.random.choice([1,2,3], p=[0.7,0.2,0.1])#若要mask，进行x_gram mask的概率
                if ngram==3 and len(rands)<7:#太大的gram不要应用于过短文本
                    ngram=2
                if ngram==2 and len(rands)<4:
                    ngram=1

                L=idx+1
                R=idx+ngram#最终需要mask的右边界（开）
                while L<R and L<len(rands):
                    rands[L]=np.random.random()*0.15#强制mask
                    L+=1
                idx=R
                if idx<len(rands):
                    rands[idx]=1#禁止mask片段的下一个token被mask，防止一大片连续mask
            idx+=1

        for r, i in zip(rands, text_ids):
            if r < 0.15 * 0.8:
                input_ids.append(self.tk.mask_token_id)
                output_ids.append(i)#mask预测自己
            elif r < 0.15 * 0.9:
                input_ids.append(i)
                output_ids.append(i)#自己预测自己
            elif r < 0.15:
                input_ids.append(np.random.randint(self.spNum,self.tkNum))
                output_ids.append(i)#随机的一个词预测自己，随机词不会从特殊符号中选取，有小概率抽到自己
            else:
                input_ids.append(i)
                output_ids.append(-100)#保持原样不预测

        return input_ids, output_ids

    #耗时操作在此进行，可用上多进程
    def __getitem__(self, item):
        text=self.data[item]#预处理，mask等操作

        text=truncate(text,self.maxLen)#截断
        text=self.tk.tokenize(text)
        text_ids = self.tk.convert_tokens_to_ids(text)

        text_ids, out_ids = self.random_mask(text_ids)#添加mask预测

        input_ids = [self.tk.cls_token_id] + text_ids + [self.tk.sep_token_id]#输入的input_id
        token_type_ids=[0]*(len(text_ids)+2)# token 对应的句子id，值为0或1(0表示对应的token属于第一句，1表示属于第二句)

        labels = [-100] + out_ids + [-100] #目标id
        assert len(input_ids)==len(token_type_ids)==len(labels)
        return {'input_ids':input_ids,'token_type_ids':token_type_ids,'labels':labels}

    @classmethod
    def collate(cls,batch):
        input_ids=[i['input_ids'] for i in batch]
        token_type_ids=[i['token_type_ids'] for i in batch]
        labels=[i['labels'] for i in batch]

        input_ids=paddingList(input_ids,0,returnTensor=True)
        token_type_ids=paddingList(token_type_ids,0,returnTensor=True)
        labels=paddingList(labels,-100,returnTensor=True)
        attention_mask=(input_ids!=0)

        return {'input_ids':input_ids,'token_type_ids':token_type_ids
                ,'attention_mask':attention_mask,'labels':labels}



##
unionList=lambda ls:list(chain(*ls))#按元素拼接
splitList=lambda x,bs:[x[i:i+bs] for i in range(0,len(x),bs)]#按bs切分

#sortBsNum：原序列按多少个bs块为单位排序，可用来增强随机性
#比如如果每次打乱后都全体一起排序，那每次都是一样的
def blockShuffle(data:list,bs:int,sortBsNum,key):

    random.shuffle(data)#先打乱
    tail=len(data)%bs#计算碎片长度
    tail=[] if tail==0 else data[-tail:]
    data=data[:len(data)-len(tail)]

    assert len(data)%bs==0#剩下的一定能被bs整除

    sortBsNum=len(data)//bs if sortBsNum is None else sortBsNum#为None就是整体排序
    data=splitList(data,sortBsNum*bs)
    data=[sorted(i,key=key,reverse=True) for i in data]#每个大块进行降排序
    data=unionList(data)
    data=splitList(data,bs)#最后，按bs分块
    random.shuffle(data)#块间打乱
    data=unionList(data)+tail
    return data
##
from torch.utils.data.dataloader import _SingleProcessDataLoaderIter,_MultiProcessingDataLoaderIter
#每轮迭代重新分块shuffle数据的DataLoader
class blockShuffleDataLoader(DataLoader):
    def __init__(self, dataset: Dataset,sortBsNum,key,**kwargs):
        assert isinstance(dataset.data,list)#需要有list类型的data属性
        super().__init__(dataset,**kwargs)#父类的参数传过去
        self.sortBsNum=sortBsNum
        self.key=key

    def __iter__(self):
        #分块shuffle
        self.dataset.data=blockShuffle(self.dataset.data,self.batch_size,self.sortBsNum,self.key)
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIter(self)
        else:
            return _MultiProcessingDataLoaderIter(self)
