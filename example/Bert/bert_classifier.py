import numpy as np
import wandb
import pandas as pd
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from model.ELMO import ELMOClassificationModel
import torch.nn.utils.rnn as rnn_utils
import torch
from torch.utils.data import Dataset
from transformers import BertModel, BertTokenizer


if __name__ == '__main__':
    # 定义路径
    data_path = '../../data/ChineseNlpCorpus/_processed.csv'
    data_path = '../data/ChineseNlpCorpus_processed.csv'
    stopwords_path = '../data/stopwords.txt'
    model_path = '../word2vec/word2vec.pkl'
    # 数据导入
    data = pd.read_csv(data_path)
    # 切分训练集和验证集、测试集
    X = data['review']  # 形如：pd.DataFrame(['我 是 一个 好人', '你 是 一个 坏人'])
    y = data['label']  # 形如：pd.DataFrame([1,0])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)

    # bert模型导入
    model_name = 'bert-large-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name) # 加载分词器
    model = BertModel.from_pretrained(model_name)  # 加载模型

    a = tokenizer.batch_encode_plus(X_train, max_length=512, padding='max_length', truncation=True, return_tensors='pt')

    # 因为bert找不到对应的词向量，所以还是使用单字的词向量
    # X_train_data = []
    # for i in X:
    #     input_ids = torch.tensor(tokenizer.encode(i, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
    #     outputs = model(input_ids)  # 输出：last_hidden_states, pooler_output
    #     # last_hidden_states是最后一层的隐藏状态
    #     X_train_data.append(outputs.last_hidden_state.shape)  # torch.Size([1, 8, 1024])


