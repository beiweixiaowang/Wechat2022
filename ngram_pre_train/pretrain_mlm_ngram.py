##
import numpy as np
import random
import os
import pandas as pd

random.seed(2022)
np.random.seed(2022)  # seed应该在main里尽早设置，以防万一
os.environ['PYTHONHASHSEED'] = str(0)  # 消除hash算法的随机性
from transformers_down import Trainer, TrainingArguments, BertTokenizer, BertConfig
from pretrain_utils import MLM_Data, blockShuffleDataLoader
import warnings
# from nezha_model.model_nazha import NeZhaForMaskedLM
from transformers import BertForMaskedLM

warnings.simplefilter('ignore')
##
maxlen = 120
batch_size = 32


def loadData(path, data_type):
    df = pd.read_pickle(path)
    word = []
    if data_type == 'train':
        line = df['text'] + df['category_id'].astype('str')
        word = line.tolist()
    elif data_type == 'unlabeled':
        word = df['text'].tolist()
    return word


# ##
# #读取数据
# unlabeled_data = loadData(path='../temp_data/unlabeled_text.pkl', data_type='unlabeled')  # 1014425
train_data = loadData(path='../temp_data/train_text.pkl', data_type='train')  # 4w
test_data = loadData(path='../temp_data/test_text.pkl', data_type='unlabeled')  # 1w


# unlabeled_sample = random.sample(unlabeled_data, 450000)
train_all =  train_data + test_data
random.shuffle(train_all)
print("训练集样本数量：", len(train_all))



# ##
# 加载预训练词典
vocab_file_dir = '../pretrain_model/chinese-roberta-wwm-ext'
tokenizer = BertTokenizer.from_pretrained(vocab_file_dir)  # 加载分词器
config = BertConfig(
    vocab_size=len(tokenizer),
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    max_position_embeddings=512,
)
# ##
# # 加载模型
model = BertForMaskedLM.from_pretrained(vocab_file_dir)
model.resize_token_embeddings(len(tokenizer))
print('============================model_frame============================\n')
# print(nezha_model)
##
train_MLM_data = MLM_Data(train_all, maxlen, tokenizer)
print(train_MLM_data)
# ##
# # 自己定义dataloader，不要用huggingface的
dl = blockShuffleDataLoader(train_MLM_data, None, key=lambda x: len(x[0]) + len(x[1]), shuffle=False
                            , batch_size=batch_size, collate_fn=train_MLM_data.collate)
print(dl)
# ##
training_args = TrainingArguments(
    # 训练过程保存的权重
    output_dir='../continue_pre/ngram_roberta',
    overwrite_output_dir=True,

    num_train_epochs=6,
    per_device_train_batch_size=batch_size,

    save_steps=5000,  # 每10个epoch save一次
    save_total_limit=3,
    logging_steps=len(dl),  # 每个epoch log一次
    seed=2022,

    learning_rate=5e-5,  # 学习率
    # lr_end=1e-5,#学习率衰减的终点
    #
    # weight_decay=0.01,#get_polynomial_decay_schedule_with_warmup 预热
    # warmup_steps=int(450000*150/batch_size*0.03)
)
##
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataLoader=dl
)
##
if __name__ == '__main__':
    trainer.train()
    # 模型最终保存的权重
    trainer.save_model('../continue_pre/ngram_roberta_model')
