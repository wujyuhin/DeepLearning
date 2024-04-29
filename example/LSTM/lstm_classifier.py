import numpy as np
import wandb
import pandas as pd
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from model.ELMO import ELMOClassificationModel
import torch.nn.utils.rnn as rnn_utils
import torch
from torch.utils.data import Dataset
from pytorch_lightning.callbacks import TQDMProgressBar
import sys

class LitProgressBar(TQDMProgressBar):
    """ 自定义进度条 : 使得验证集的进度条不显示"""
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        # bar.set_description("running validation...")
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

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
    :return:  data.transpose(0, 1), torch.tensor(label)

    example:
    data_tuple = [(tensor([1,2,3]), 1), (tensor([1,2,3,4,5]), 0)]
    collate(data_tuple) = (tensor([[1,2,3,0,0], [1,2,3,4,5]]), tensor([1,0]))
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

def w2v(X):
    X_test_data = []
    for item in X:
        sentences = []
        for i in item:
            try:
                sentences.append(word2vec.wv[i])
            except:
                pass
        X_test_data.append(torch.tensor(np.array(sentences)))
    return X_test_data


if __name__ == '__main__':
    # 定义路径
    data_path = '../data/ChineseNlpCorpus_processed.csv'
    stopwords_path = '../data/stopwords.txt'
    model_path = '../word2vec/word2vec.pkl'
    # 调用gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 数据导入
    data = pd.read_csv(data_path)
    # ================================== 先word2vec再生成train_loader【推荐】 ===============================================
    X = data['review']  # 形如：pd.DataFrame(['我 是 一个 好人', '你 是 一个 坏人'])
    y = data['label']  # 形如：pd.DataFrame([1,0])
    # ========== 将每一句话每个词转换成tensor vector ==========
    word2vec = Word2Vec.load(model_path)
    # 切分训练集和验证集、测试集
    train_data = HotelDataset(w2v(X), y)
    train_data,test_data = torch.utils.data.random_split(train_data, [int(len(train_data) * 0.8), len(train_data) - int(len(train_data) * 0.8)])
    train_data,val_data = torch.utils.data.random_split(train_data, [int(len(train_data) * 0.8), len(train_data) - int(len(train_data) * 0.8)])
    train_loader = DataLoader(train_data, batch_size=64, collate_fn=collate)
    valid_loader = DataLoader(val_data, batch_size=64,collate_fn=collate)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=True, drop_last=True, collate_fn=collate)
    # ================================= python-lightning =================================================
    model = ELMOClassificationModel(input_size=128, hidden_size=64, batch_size=64, output_size=2)
    trainer = pl.Trainer(max_epochs=100,callbacks=[LitProgressBar()])
    trainer.fit(model,train_loader,valid_loader)
    trainer.test(model, test_dataloaders=test_loader)
