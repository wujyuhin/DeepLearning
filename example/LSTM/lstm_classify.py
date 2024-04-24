# 根据example/data中的train、dev、test数据集，使用ANN进行文本分类，如下：
# 导入已经编写好的model文件夹中的ANNmodel.py文件
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


class HotelDataset(Dataset):
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index]

    def __len__(self):
        return len(self.x_train)


def collate(data_tuple):
    """
    定义自己的 collate_fn 方法, 用于处理变长数据.
    :param data_tuple: data_tuple是一个列表，列表中包含 batch_size 个元组，每个元组中包含数据和标签
    :return:
    """
    # 将数据从大到小排序.

    data_tuple.sort(key=lambda x: len(x[0]), reverse=True)
    # 将自变量和因变量从元组中分离.
    data = [sq[0] for sq in data_tuple]
    label = [sq[1] for sq in data_tuple]
    # data_length = [len(sq) for sq in data]

    # 用零补充，使长度对齐
    data = rnn_utils.pad_sequence(data, batch_first=True)
    return data.transpose(0, 1), torch.tensor(label)


if __name__ == '__main__':
    # 定义路径
    data_path = '../data/ChineseNlpCorpus_processed.csv'
    stopwords_path = '../data/stopwords.txt'
    model_path = '../word2vec/word2vec.pkl'
    # 调用gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ================================== 先word2vec再生成train_loader【推荐】 ===============================================
    # 切分训练集和验证集、测试集
    data = pd.read_csv(data_path)
    X = data['review']  # 形如：pd.DataFrame(['我 是 一个 好人', '你 是 一个 坏人'])
    y = data['label']  # 形如：pd.DataFrame([1,0])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    # ========== 将每一句话每个词转换成tensor vector ==========
    word2vec = Word2Vec.load(model_path)
    X_train_data = []  # shape: (batch_size, seq_len, hidden_size)
    for item in X_train:  # item: (['我', '是', '一个', '好人'], 1)
        sentences = []
        for i in item:  # i: '我'
            try:
                sentences.append(word2vec.wv[i])  # shape: sentences = [(hidden_size,), (hidden_size,), ...]
            except:
                pass
        X_train_data.append(torch.tensor(np.array(sentences)))  # shape: X_train_data = [(seq_len, hidden_size), ...]

    train_data = HotelDataset(X_train_data, y_train)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, drop_last=True, collate_fn=collate)

    # ================================= python-lightning =================================================
    model = ELMOClassificationModel(input_size=128, hidden_size=64, batch_size=64, output_size=2)
    trainer = pl.Trainer(max_epochs=100)
    trainer.fit(model, train_loader)
